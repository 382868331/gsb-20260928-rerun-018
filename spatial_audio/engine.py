"""Continuous-time spatial rendering engine.

Design guarantees:

- Time advances continuously: trajectories are evaluated once per output
  sample, not once per processing block. All carried state (playhead,
  smoothing envelopes) is sample-accurate, so splitting the render into
  different block sizes produces bit-identical output.
- Sudden emitter deactivation is ramped out with a release envelope
  (no clicks).
- Mixed sources pass through a soft safety limiter; when limiting occurs it
  is reported in the render statistics instead of clipping silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import lfilter

from .scene import Emitter, Listener

SPEED_OF_SOUND = 343.0


@dataclass
class RenderStats:
    duration: float
    sample_rate: int
    pre_limiter_peak: float
    post_limiter_peak: float
    max_gain_reduction_db: float
    limited: bool
    emitter_peaks: dict = field(default_factory=dict)


@dataclass
class RenderResult:
    samples: np.ndarray  # shape (n, 2), float64
    sample_rate: int
    stats: RenderStats

    def peak(self):
        return float(np.max(np.abs(self.samples)))


class _EmitterRuntime:
    """Per-emitter carried state, kept across blocks."""

    def __init__(self, emitter):
        self.emitter = emitter
        self.playhead = 0.0       # fractional position in source samples
        self.env_state = 0.0      # activation gate envelope
        self.gain_l_state = 0.0   # smoothed geometry gain (left)
        self.gain_r_state = 0.0   # smoothed geometry gain (right)
        self.finished = False


def _resample(resource_samples, playhead, rates, loop):
    """Linear resampling on a fractional, per-sample-rate grid.

    rates[i] is source samples per output sample (Doppler factor plus
    sample-rate ratio). Returns (mono signal, next_playhead).
    """
    increments = np.concatenate(([0.0], np.cumsum(rates[:-1])))
    positions = playhead + increments
    n_src = len(resource_samples)

    if loop and n_src > 1:
        positions = np.mod(positions, n_src)
        idx0 = np.floor(positions).astype(np.int64) % n_src
        idx1 = (idx0 + 1) % n_src
        frac = positions - np.floor(positions)
        out = resource_samples[idx0] * (1.0 - frac) + resource_samples[idx1] * frac
    else:
        valid = positions < n_src - 1
        safe = np.clip(positions, 0.0, max(n_src - 1.0001, 0.0))
        idx0 = np.floor(safe).astype(np.int64)
        idx1 = np.minimum(idx0 + 1, n_src - 1)
        frac = safe - idx0
        out = np.where(
            valid,
            resource_samples[idx0] * (1.0 - frac) + resource_samples[idx1] * frac,
            0.0,
        )
    return out, playhead + float(np.sum(rates))


class SpatialEngine:
    def __init__(
        self,
        listener,
        sample_rate=48000,
        speed_of_sound=SPEED_OF_SOUND,
        reference_distance=1.0,
        gate_attack_ms=3.0,
        gate_release_ms=15.0,
        smooth_ms=2.0,
        limit_threshold=0.95,
    ):
        self.listener = listener
        self.sample_rate = int(sample_rate)
        self.c = float(speed_of_sound)
        self.reference_distance = float(reference_distance)
        dt = 1.0 / self.sample_rate
        self._attack_coef = float(1.0 - np.exp(-dt / (gate_attack_ms / 1000.0)))
        self._release_coef = float(1.0 - np.exp(-dt / (gate_release_ms / 1000.0)))
        self._smooth_coef = float(1.0 - np.exp(-dt / (smooth_ms / 1000.0)))
        self.limit_threshold = float(limit_threshold)
        self.emitters = []
        self._runtimes = {}
        # emitter id -> list of (time, active), sample-accurate scheduling
        self._events = {}

    def add_emitter(self, emitter):
        if emitter not in self.emitters:
            self.emitters.append(emitter)
            self._runtimes[id(emitter)] = _EmitterRuntime(emitter)
        return emitter

    def remove_emitter(self, emitter):
        if emitter in self.emitters:
            self.emitters.remove(emitter)
            self._runtimes.pop(id(emitter), None)

    def schedule_active(self, emitter, time_seconds, active):
        """Schedule activation/deactivation at a precise continuous time."""
        self._events.setdefault(id(emitter), []).append(
            (float(time_seconds), bool(active))
        )

    def _listener_pose(self, times):
        traj = getattr(self.listener, "trajectory", None)
        if traj is not None:
            return traj.sample(times)
        pos = np.broadcast_to(self.listener.position, (len(times), 3)).copy()
        return pos, np.zeros_like(pos)

    def _gate(self, emitter, rt, block_start, times):
        """Sample-accurate activation gate with fast attack / slow release."""
        n = len(times)
        changes = sorted(self._events.get(id(emitter), []))
        base = 1.0 if emitter.active else 0.0
        for te, active in changes:
            if te <= block_start:
                base = 1.0 if active else 0.0
        target = np.full(n, base)
        for te, active in changes:
            target = np.where(times >= te, 1.0 if active else 0.0, target)

        edges = {0, n}
        for te, _ in changes:
            k = int(np.searchsorted(times, te, side="left"))
            if 0 < k < n:
                edges.add(k)
        edges = sorted(edges)

        gate = np.empty(n)
        state = rt.env_state
        for a, b in zip(edges[:-1], edges[1:]):
            active = target[a] > 0.5
            coef = self._attack_coef if active else self._release_coef
            seg, zi = lfilter(
                [coef], [1.0, -(1.0 - coef)],
                np.full(b - a, float(target[a])), zi=[state],
            )
            gate[a:b] = seg
            state = float(zi[0])
        rt.env_state = state
        return gate

    def render_block(self, time_seconds, num_samples):
        """Render one block starting at continuous time time_seconds."""
        n = int(num_samples)
        times = time_seconds + (np.arange(n) + 0.5) / self.sample_rate

        lpos, lvel = self._listener_pose(times)
        fwd = self.listener.forward
        right = self.listener.right

        mixed = np.zeros((n, 2), dtype=np.float64)
        peaks = {}
        sb = [self._smooth_coef]
        sa = [1.0, -(1.0 - self._smooth_coef)]

        for emitter in self.emitters:
            rt = self._runtimes[id(emitter)]
            epos, evel = emitter.trajectory.sample(times)
            rel = epos - lpos  # unit vector: listener -> source
            dist = np.linalg.norm(rel, axis=1)
            rhat = rel / np.maximum(dist, 1e-9)[:, None]

            # Doppler along the listener->source axis: source approaching
            # means evel . rhat < 0 (rate > 1, pitch rises).
            rate = (self.c + np.sum(lvel * rhat, axis=1)) / (
                self.c + np.sum(evel * rhat, axis=1)
            )
            rate = np.clip(rate, 0.25, 4.0)
            if emitter.resource.sample_rate != self.sample_rate:
                rate *= emitter.resource.sample_rate / self.sample_rate

            src, rt.playhead = _resample(
                emitter.resource.samples, rt.playhead, rate, emitter.loop
            )
            if not emitter.loop and rt.playhead >= len(emitter.resource.samples):
                rt.finished = True

            dist_gain = 1.0 / (1.0 + dist / self.reference_distance)

            # Equal-power pan from angle to the listener's right axis.
            x = np.clip(np.sum(rhat * right, axis=1), -1.0, 1.0)
            gl = np.sqrt(np.clip(0.5 * (1.0 - x), 0.0, 1.0))
            gr = np.sqrt(np.clip(0.5 * (1.0 + x), 0.0, 1.0))
            # Mild front/back cue.
            frontness = np.clip(np.sum(rhat * fwd, axis=1), 0.0, 1.0)
            rear_gain = 0.85 + 0.15 * frontness

            gain_l = dist_gain * gl * rear_gain * emitter.volume
            gain_r = dist_gain * gr * rear_gain * emitter.volume

            gl_s, zi = lfilter(sb, sa, gain_l, zi=[rt.gain_l_state])
            rt.gain_l_state = float(zi[0])
            gr_s, zi = lfilter(sb, sa, gain_r, zi=[rt.gain_r_state])
            rt.gain_r_state = float(zi[0])

            gate = self._gate(emitter, rt, time_seconds, times)

            ch_l = src * gl_s * gate
            ch_r = src * gr_s * gate
            mixed[:, 0] += ch_l
            mixed[:, 1] += ch_r
            peaks[emitter.name] = max(
                peaks.get(emitter.name, 0.0),
                float(np.max(np.abs(ch_l))),
                float(np.max(np.abs(ch_r))),
            )
        return mixed, peaks

    def _limit(self, stereo):
        threshold = self.limit_threshold
        if not stereo.size:
            return stereo, 0.0
        # Transparent at and below the threshold; tanh knee only above it.
        over = np.abs(stereo) > threshold
        limited = np.where(
            over,
            threshold * np.tanh(stereo / threshold),
            stereo,
        )
        reduction = float(np.max(np.abs(stereo) - np.abs(limited))) if stereo.size else 0.0
        return limited, reduction

    def render(self, duration, block_size=1024):
        """Render duration seconds; block_size never changes the result."""
        n_total = int(round(duration * self.sample_rate))
        chunks = []
        emitter_peaks = {}
        pos, t0 = 0, 0.0
        while pos < n_total:
            n = min(int(block_size), n_total - pos)
            block, peaks = self.render_block(t0, n)
            chunks.append(block)
            for name, pk in peaks.items():
                emitter_peaks[name] = max(emitter_peaks.get(name, 0.0), pk)
            pos += n
            t0 += n / self.sample_rate

        raw = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2))
        pre_peak = float(np.max(np.abs(raw))) if raw.size else 0.0
        limited, max_red = self._limit(raw)
        post_peak = float(np.max(np.abs(limited))) if limited.size else 0.0
        stats = RenderStats(
            duration=float(duration),
            sample_rate=self.sample_rate,
            pre_limiter_peak=pre_peak,
            post_limiter_peak=post_peak,
            max_gain_reduction_db=(
                float(20.0 * np.log10(pre_peak / max(post_peak, 1e-12)))
                if max_red > 1e-12 else 0.0
            ),
            limited=max_red > 1e-12,
            emitter_peaks=emitter_peaks,
        )
        return RenderResult(samples=limited, sample_rate=self.sample_rate, stats=stats)
