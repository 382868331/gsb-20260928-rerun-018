from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .listener import Listener
from .source import SoundSource

SPEED_OF_SOUND = 343.0
EAR_HALF_SPACING = 0.09
MAX_ILD_DB = 6.0


@dataclass
class RenderReport:
    peak_pre_limiter: float = 0.0
    over_level_events: int = 0
    stopped_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def overflowed(self) -> bool:
        """True if the un-limited mix would have clipped the 1.0 ceiling."""
        return self.peak_pre_limiter > 1.0


class _EarDelayLine:
    """Interpolating circular delay line; variable delay gives Doppler."""

    def __init__(self, sample_rate: int, max_seconds: float = 2.0):
        self.size = int(sample_rate * max_seconds) + 4
        self.buf = np.zeros(self.size, dtype=np.float64)
        self.write_index = 0

    def write_block(self, block: np.ndarray) -> None:
        n = len(block)
        end = self.write_index + n
        if end <= self.size:
            self.buf[self.write_index:end] = block
        else:
            first = self.size - self.write_index
            self.buf[self.write_index:] = block[:first]
            self.buf[: n - first] = block[first:]
        self.write_index = (self.write_index + n) % self.size

    def read_delayed(self, delays: np.ndarray) -> np.ndarray:
        n = len(delays)
        # write_index points one past the newest sample; sample k in the block
        # is at write_index - n + k, so read position k is:
        read_positions = (self.write_index - n + np.arange(n) - delays) % self.size
        i0 = np.floor(read_positions).astype(np.int64) % self.size
        frac = read_positions - np.floor(read_positions)
        i1 = (i0 + 1) % self.size
        return self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac


class _SourceRuntime:
    def __init__(self, source: SoundSource, sample_rate: int, attack_s: float, release_s: float):
        self.source = source
        self.lines = [_EarDelayLine(sample_rate), _EarDelayLine(sample_rate)]
        self.attack = 1.0 / max(1.0, attack_s * sample_rate)
        self.release = 1.0 / max(1.0, release_s * sample_rate)
        self.env = 0.0
        self.was_active = source.active
        self.tail = False
        self.stopped = False
        self.generated = 0.0  # seconds of signal already generated

    @property
    def alive(self) -> bool:
        return (not self.tail) or self.env > 1e-5


class SpatialEngine:
    """Continuous-time stereo spatial audio renderer.

    All geometry is evaluated per audio sample, so requesting different block
    sizes from the streaming API only changes how the same output is split.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        listener: Listener | None = None,
        sources: list[SoundSource] | None = None,
        ref_distance: float = 2.0,
        attack_ms: float = 5.0,
        release_ms: float = 30.0,
        headroom: float = 1.0,
    ):
        self.sample_rate = int(sample_rate)
        self.listener = listener or Listener()
        self.sources: list[SoundSource] = list(sources or [])
        self.ref_distance = float(ref_distance)
        self.headroom = float(headroom)
        self.time = 0.0
        self.report = RenderReport()
        self._runtimes: dict[int, _SourceRuntime] = {}
        self._attack = attack_ms / 1000.0
        self._release = release_ms / 1000.0

    def add_source(self, source: SoundSource) -> None:
        self.sources.append(source)

    def render(self, duration: float) -> np.ndarray:
        """Render the whole interval in one go (used for export)."""
        n = int(round(duration * self.sample_rate))
        return self.process(n)

    def process(self, n_samples: int) -> np.ndarray:
        # DSP always advances in fixed 512-sample sub-blocks consumed from a
        # persistent FIFO, so the caller may request any chunk size without
        # re-running per-sample setup. Block boundaries never alter samples.
        if not hasattr(self, "_queue"):
            from collections import deque
            self._queue: deque[np.ndarray] = deque()
            self._queue_pos = 0
        parts: list[np.ndarray] = []
        collected = 0
        while collected < n_samples:
            if not self._queue:
                self._queue.append(self._process_block(512))
                self._queue_pos = 0
            block = self._queue[0]
            take = min(n_samples - collected, len(block) - self._queue_pos)
            parts.append(block[self._queue_pos:self._queue_pos + take])
            self._queue_pos += take
            collected += take
            if self._queue_pos >= len(block):
                self._queue.popleft()
                self._queue_pos = 0
        return np.concatenate(parts, axis=0)

    def _runtime_for(self, source: SoundSource) -> _SourceRuntime:
        rt = self._runtimes.get(id(source))
        if rt is None:
            rt = _SourceRuntime(source, self.sample_rate, self._attack, self._release)
            self._runtimes[id(source)] = rt
        return rt

    def _process_block(self, n: int) -> np.ndarray:
        t0 = self.time
        t = t0 + (np.arange(n, dtype=np.float64) + 0.5) / self.sample_rate
        f, r_axis, _ = self._basis()
        lp = self.listener.position.as_array()
        ears = [lp - EAR_HALF_SPACING * r_axis, lp + EAR_HALF_SPACING * r_axis]

        mix = np.zeros((n, 2), dtype=np.float64)
        finished: list[_SourceRuntime] = []

        for source in self.sources:
            rt = self._runtime_for(source)
            if source.signal is None or not rt.alive:
                continue

            # Emitted signal, generated on a continuous time axis.
            dry = source.signal.render(rt.generated, n, self.sample_rate)
            rt.generated += n / self.sample_rate

            # Envelope (sample-accurate, state carried across blocks). A
            # deactivate_at event or an external update() switches the target
            # at an exact sample, so the 30 ms release starts precisely then.
            scheduled = np.ones(n, dtype=bool)
            if source.deactivate_at is not None:
                scheduled &= t < source.deactivate_at
            if not source.active:
                scheduled[:] = False
            if source.deactivate_at is not None and self.time >= source.deactivate_at:
                rt.stopped = True
            # Externally re-enabling (active=True, no past schedule) restarts it.
            if rt.stopped and source.active and (
                source.deactivate_at is None or source.deactivate_at > t[-1]
            ):
                rt.stopped = False
                rt.tail = False
            now_active = bool(scheduled[-1]) if n else rt.was_active
            if rt.was_active and not now_active and not rt.stopped:
                rt.tail = True
                rt.stopped = bool(source.deactivate_at is not None)
                self.report.stopped_sources.append(source.name)
                self.report.notes.append(
                    f"source '{source.name}' deactivated at t={float(self.time):.3f}s; fade-out started"
                )
            rt.was_active = now_active

            # Piecewise-linear attack/release, vectorized over constant-target runs.
            want = scheduled & (not rt.tail) & (not rt.stopped)
            change = np.flatnonzero(np.diff(want.astype(np.int8))) + 1
            bounds = [0, *change.tolist(), n]
            env = np.empty(n, dtype=np.float64)
            e = rt.env
            for a, b in zip(bounds[:-1], bounds[1:]):
                k = np.arange(1, b - a + 1)
                if want[a]:
                    e_new = np.minimum(1.0, e + rt.attack * k)
                else:
                    e_new = np.maximum(0.0, e - rt.release * k)
                env[a:b] = e_new
                e = float(e_new[-1])
            rt.env = e

            dry = dry * env * source.gain
            for ear_idx in (0, 1):
                rt.lines[ear_idx].write_block(dry)

            # Per-sample geometry for the moving source.
            positions, _velocities = self._sample_motion(source, t)

            for ear_idx in (0, 1):
                rel = positions - ears[ear_idx][None, :]
                dist = np.linalg.norm(rel, axis=1)
                direction = rel / np.maximum(dist, 1e-9)[:, None]
                side = direction @ r_axis  # +1 toward this ear, -1 away

                delay_s = dist / SPEED_OF_SOUND
                delays = np.maximum(delay_s * self.sample_rate, 0.0)

                distance_gain = self.ref_distance / (self.ref_distance + dist)
                # Ear 0 sits on the -right axis; for it a negative side means
                # the source is on this ear's side (no shadow).
                near_sign = -1.0 if ear_idx == 0 else 1.0
                shadow = np.clip(-near_sign * side, 0.0, 1.0)
                ild = 10.0 ** (-MAX_ILD_DB * shadow / 20.0)
                gain = distance_gain * ild

                audio = rt.lines[ear_idx].read_delayed(delays)
                mix[:, ear_idx] += gain * audio

            if rt.tail and rt.env <= 0.0:
                finished.append(rt)

        for rt in finished:
            self._runtimes.pop(id(rt.source), None)

        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        self.report.peak_pre_limiter = max(self.report.peak_pre_limiter, peak)
        over = int(np.count_nonzero(np.abs(mix) > self.headroom))
        self.report.over_level_events += over
        if over:
            self.report.notes.append(
                f"{over} samples exceeded headroom {self.headroom:.2f}; soft-limited (no clipping)"
            )

        self.time += n / self.sample_rate
        # Soft limiter guarantees the mixed result never silently overflows [-1, 1].
        limited = np.tanh(mix / self.headroom) * self.headroom
        return limited

    def _basis(self):
        cur = (self.listener.orientation.forward, self.listener.orientation.up)
        if getattr(self, "_basis_key", None) != cur:
            self._basis_cache = self.listener.orientation.basis()
            self._basis_key = cur
        return self._basis_cache
    @staticmethod
    def _sample_motion(source: SoundSource, t: np.ndarray):
        traj = source.trajectory
        if traj is None:
            p = source.position.as_array()
            return np.broadcast_to(p, (len(t), 3)).copy(), np.zeros((len(t), 3))
        if hasattr(traj, "t") and hasattr(traj, "p"):
            positions = np.empty((len(t), 3))
            for axis in range(3):
                positions[:, axis] = np.interp(t, traj.t, traj.p[:, axis])
            idx = np.clip(np.searchsorted(traj.t, t, side="right") - 1, 0, len(traj.t) - 2)
            dt = np.diff(traj.t)
            safe_dt = np.where(dt == 0, 1.0, dt)
            slopes = np.zeros_like(traj.p)
            slopes[:-1] = np.diff(traj.p, axis=0) / safe_dt[:, None]
            velocities = slopes[np.minimum(idx, len(slopes) - 1)]
            return positions, velocities
        positions = np.array([traj.sample(float(ti))[0] for ti in t])
        return positions, np.zeros_like(positions)
