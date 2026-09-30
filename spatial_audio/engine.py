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
from .rooms import RoomReverb

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
        self.lp_state = 0.0     # occlusion lowpass filter state
        self.occl_state = 0.0   # smoothed occlusion (muffle) mix


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
        scene=None,
        occlusion_cutoff=700.0,
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
        self.scene = scene
        a0 = float(1.0 - np.exp(-2.0 * np.pi * occlusion_cutoff / self.sample_rate))
        self._lp_b = [a0]
        self._lp_a = [1.0, -(1.0 - a0)]
        if scene is not None:
            self._reverbs = [
                RoomReverb(self.sample_rate, room.reverb_time)
                for room in scene.rooms
            ]
            self._reverb_gain_states = [0.0] * len(scene.rooms)
        else:
            self._reverbs = []
            self._reverb_gain_states = []
        self.emitters = []
        self._runtimes = {}
        # emitter id -> list of (time, active), sample-accurate scheduling
        self._events = {}
        self._current_time = 0.0
        # Integer index of the current engine clock within the take:
        # sample-centered time is (clock_sample + 0.5) / sample_rate.
        # Keeping the index exactly integer makes a render split at an
        # arbitrary number of samples bit-identical to one continuous
        # render regardless of block size.
        self._clock_sample = 0

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

    @property
    def current_time(self):
        """Continuous time up to which this engine has rendered."""
        return self._current_time

    def suspend_state(self):
        """Snapshot every piece of carried render state.

        The snapshot is JSON-serialisable (plain lists) and lets a render
        be stopped mid-way and resumed on an identically constructed
        engine with sample-exact output; reverb delay lines are included,
        so tails that have not decayed yet are never erased.
        """
        emitters_state = []
        for emitter in self.emitters:
            rt = self._runtimes[id(emitter)]
            emitters_state.append({
                "name": emitter.name,
                "playhead": rt.playhead,
                "env_state": rt.env_state,
                "gain_l_state": rt.gain_l_state,
                "gain_r_state": rt.gain_r_state,
                "lp_state": rt.lp_state,
                "occl_state": rt.occl_state,
                "finished": rt.finished,
            })
        return {
            "time": self._current_time,
            "clock_sample": self._clock_sample,
            "emitters": emitters_state,
            "reverb_gain_states": list(self._reverb_gain_states),
            "reverbs": [r.get_state() for r in self._reverbs],
        }

    def restore_state(self, state):
        """Restore a snapshot from :meth:`suspend_state`.

        Emitters are matched by name; the engine must be rebuilt with the
        same scene and emitters (e.g. via ``build_engine`` of the same
        recording) before restoring.
        """
        by_name = {e.name: e for e in self.emitters}
        for es in state["emitters"]:
            emitter = by_name.get(es["name"])
            if emitter is None:
                raise ValueError(f"unknown emitter in saved state: {es['name']}")
            rt = self._runtimes[id(emitter)]
            rt.playhead = float(es["playhead"])
            rt.env_state = float(es["env_state"])
            rt.gain_l_state = float(es["gain_l_state"])
            rt.gain_r_state = float(es["gain_r_state"])
            rt.lp_state = float(es["lp_state"])
            rt.occl_state = float(es["occl_state"])
            rt.finished = bool(es["finished"])
        if len(state.get("reverbs", [])) != len(self._reverbs):
            raise ValueError("reverb state does not match current scene")
        for reverb, rs in zip(self._reverbs, state["reverbs"]):
            reverb.set_state(rs)
        gains = state.get("reverb_gain_states", [])
        if len(gains) != len(self._reverb_gain_states):
            raise ValueError("reverb gain state does not match current scene")
        self._reverb_gain_states = [float(g) for g in gains]
        self._clock_sample = int(state.get(
            "clock_sample",
            round(float(state["time"]) * self.sample_rate + 0.5),
        ))
        self._current_time = (self._clock_sample - 0.5) / self.sample_rate
        return self

    def save_state(self, path):
        """Write :meth:`suspend_state` to a JSON file."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.suspend_state(), f)

    def load_state(self, path):
        """Read a state file written by :meth:`save_state`."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            return self.restore_state(json.load(f))

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
        times = time_seconds + np.arange(n) / self.sample_rate

        lpos, lvel = self._listener_pose(times)
        fwd = self.listener.forward
        right = self.listener.right

        mixed = np.zeros((n, 2), dtype=np.float64)
        scene = self.scene
        if scene is not None:
            listener_rooms = scene.room_index_at(lpos)
            room_inputs = [np.zeros(n) for _ in scene.rooms]
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

            if scene is not None:
                source_rooms = scene.room_index_at(epos)
                trans, muffle = scene.direct_transmission(
                    listener_rooms, source_rooms, times
                )
                gain_l = gain_l * trans
                gain_r = gain_r * trans
                # Muffled copy through a fixed lowpass; the smoothed mix
                # crossfades dry/muffled so the spectrum moves continuously.
                muffled, lp_zf = lfilter(
                    self._lp_b, self._lp_a, src, zi=[rt.lp_state]
                )
                rt.lp_state = float(lp_zf[0])
                mix_s, oc_zf = lfilter(
                    sb, sa, muffle, zi=[rt.occl_state]
                )
                rt.occl_state = float(oc_zf[0])
                src = src * (1.0 - mix_s) + muffled * mix_s

            gl_s, zi = lfilter(sb, sa, gain_l, zi=[rt.gain_l_state])
            rt.gain_l_state = float(zi[0])
            gr_s, zi = lfilter(sb, sa, gain_r, zi=[rt.gain_r_state])
            rt.gain_r_state = float(zi[0])

            gate = self._gate(emitter, rt, time_seconds, times)

            if scene is not None:
                send_base = src * gate * dist_gain * emitter.volume
                for ri, room in enumerate(scene.rooms):
                    m = source_rooms == ri
                    if m.any():
                        room_inputs[ri][m] += send_base[m] * room.reverb_send

            ch_l = src * gl_s * gate
            ch_r = src * gr_s * gate
            mixed[:, 0] += ch_l
            mixed[:, 1] += ch_r
            peaks[emitter.name] = max(
                peaks.get(emitter.name, 0.0),
                float(np.max(np.abs(ch_l))),
                float(np.max(np.abs(ch_r))),
            )

        if scene is not None:
            for ri, room in enumerate(scene.rooms):
                wet = self._reverbs[ri].process(room_inputs[ri])
                target = scene.reverb_transmission(listener_rooms, ri, times)
                g, zi = lfilter(
                    sb, sa, target, zi=[self._reverb_gain_states[ri]]
                )
                self._reverb_gain_states[ri] = float(zi[0])
                mixed += wet * g[:, None]
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

    def render(self, duration, block_size=1024, start_time=None):
        """Render ``duration`` seconds and advance the engine clock.

        Without ``start_time`` rendering continues from the engine clock
        (0 for a fresh engine, or the sample reached by previous /
        restored blocks), so a render split into several calls with a
        saved state in between is sample-exact with one continuous
        render, independent of the block size used for either half.
        ``start_time`` (seconds) may instead jump the clock; it is rounded
        to the nearest sample boundary.
        """
        n_total = int(round(duration * self.sample_rate))
        if start_time is not None:
            self._clock_sample = int(round(float(start_time) * self.sample_rate))
            self._current_time = (self._clock_sample - 0.5) / self.sample_rate
        chunks = []
        emitter_peaks = {}
        pos = 0
        # Every sample is timed from an exact integer sample index into
        # the take (time = (k + 0.5) / sample_rate), never by accumulating
        # float block durations, so split and continuous renders share the
        # exact same sample times and recursive filters.
        while pos < n_total:
            n = min(int(block_size), n_total - pos)
            k0 = self._clock_sample + pos
            block_t0 = (k0 + 0.5) / self.sample_rate
            block, peaks = self.render_block(block_t0, n)
            chunks.append(block)
            for name, pk in peaks.items():
                emitter_peaks[name] = max(emitter_peaks.get(name, 0.0), pk)
            pos += n
        self._clock_sample += pos
        # Clock stores the centered time of the last rendered sample.
        self._current_time = (self._clock_sample - 0.5) / self.sample_rate

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
