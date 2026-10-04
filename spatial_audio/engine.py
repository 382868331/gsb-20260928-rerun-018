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
        # Per-crossover lowpass states for frequency-band splitting of the
        # direct path and of the reverb send (lazily sized to the scene).
        self.direct_band_states = None
        self.send_band_states = None


def _split_bands(x, filters, states):
    """Split mono x into len(filters)+1 complementary bands.

    Each stage lowpasses the residual of the previous stage, so the bands
    sum back to the input. states carries one zi float per stage across
    blocks, keeping the split sample-exact for any block size.
    """
    bands = []
    rest = x
    for k, (b, a) in enumerate(filters):
        low, zf = lfilter(b, a, rest, zi=[states[k]])
        states[k] = float(zf[0])
        bands.append(low)
        rest = rest - low
    bands.append(rest)
    return bands


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
            # One-pole crossovers splitting the signal into the scene
            # frequency bands (only used when a zone has per-band gains).
            self._band_filters = []
            for edge in scene.band_edges:
                a0 = float(1.0 - np.exp(-2.0 * np.pi * edge / self.sample_rate))
                self._band_filters.append(([a0], [1.0, -(1.0 - a0)]))
        else:
            self._reverbs = []
            self._reverb_gain_states = []
            self._band_filters = []
        self.emitters = []
        self._runtimes = {}
        # emitter id -> list of (time, active), sample-accurate scheduling
        self._events = {}
        # Session clock for chunked rendering / checkpoints. render() always
        # starts from t=0 on fresh state; render_chunk() advances this clock.
        self._clock = 0.0

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
                    listener_rooms, source_rooms, times, lpos, epos
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

            direct_src = src
            if scene is not None and scene.has_band_processing:
                # Frequency-dependent absorption: split into bands and
                # weigh each by the zones the direct paths cross. Sound
                # produced now uses the materials in place now; tails
                # already inside the reverb buses are untouched.
                if rt.direct_band_states is None:
                    rt.direct_band_states = [0.0] * len(self._band_filters)
                mult = scene.direct_zone_bands(
                    listener_rooms, source_rooms, times, lpos, epos
                )
                bands = _split_bands(src, self._band_filters, rt.direct_band_states)
                direct_src = np.zeros_like(src)
                for bi, band in enumerate(bands):
                    direct_src += band * mult[:, bi]

            gl_s, zi = lfilter(sb, sa, gain_l, zi=[rt.gain_l_state])
            rt.gain_l_state = float(zi[0])
            gr_s, zi = lfilter(sb, sa, gain_r, zi=[rt.gain_r_state])
            rt.gain_r_state = float(zi[0])

            gate = self._gate(emitter, rt, time_seconds, times)

            if scene is not None:
                send_base = src * gate * dist_gain * emitter.volume
                if scene.has_band_processing:
                    # Band-filter the send where it enters the room bus,
                    # so a decaying tail keeps the spectrum of the
                    # materials present when it was produced.
                    if rt.send_band_states is None:
                        rt.send_band_states = [0.0] * len(self._band_filters)
                    send_mult = scene.send_band_factors(
                        epos, source_rooms, times
                    )
                    send_bands = _split_bands(
                        send_base, self._band_filters, rt.send_band_states
                    )
                    send_base = np.zeros_like(send_base)
                    for bi, band in enumerate(send_bands):
                        send_base += band * send_mult[:, bi]
                for ri, room in enumerate(scene.rooms):
                    m = source_rooms == ri
                    if m.any():
                        room_inputs[ri][m] += send_base[m] * room.reverb_send

            ch_l = direct_src * gl_s * gate
            ch_r = direct_src * gr_s * gate
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
                target = scene.reverb_transmission(
                    listener_rooms, ri, times, lpos
                )
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

    def _render_span(self, start_time, duration, block_size):
        """Render [start_time, start_time+duration) on the carried state."""
        n_total = int(round(duration * self.sample_rate))
        chunks = []
        emitter_peaks = {}
        pos, t0 = 0, float(start_time)
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

    def render(self, duration, block_size=1024):
        """Render duration seconds from t=0; block_size never changes the result."""
        return self._render_span(0.0, duration, block_size)

    def render_chunk(self, duration, block_size=1024):
        """Render the next chunk of the session, advancing the clock.

        All carried state (playheads, envelopes, reverb tails) continues
        across calls, so consecutive chunks concatenate to the same signal
        as one long render.
        """
        result = self._render_span(self._clock, duration, block_size)
        self._clock += float(duration)
        return result

    def render_interval(self, start, end, block_size=1024):
        """Render [start, end); identical to slicing a full render from 0.

        The engine fast-forwards its state through [0, start) first, so
        reverb tails and envelopes at the interval start match the full
        render exactly.
        """
        start = float(start)
        end = float(end)
        if not 0.0 <= start < end:
            raise ValueError("expected 0 <= start < end")
        if start > 0.0:
            self._render_span(0.0, start, block_size)
        return self._render_span(start, end - start, block_size)

    def reset_reverb_buses(self):
        """Discard reverb tail state, keeping direct-path carry state.

        This is the boundary-recompute choice: after fast-forwarding to a
        revision boundary, the direct sound and its envelopes continue
        from the actual history, but no pre-boundary reverb tail is kept.
        The default render instead keeps every tail already produced.
        """
        for reverb in self._reverbs:
            reverb.reset_state()
        self._reverb_gain_states = [0.0] * len(self._reverb_gain_states)

    def render_recomputed_from(self, boundary, duration,
                               block_size=1024, reset_reverbs=True):
        """Recompute the tail of a take starting at a revision boundary.

        Fast-forward [0, boundary) so direct-path envelopes and per-source
        filters remain physically continuous, optionally discard the reverb
        buses at the boundary, then render [boundary, boundary+duration).
        With reset_reverbs=False this is the ordinary history-preserving
        render; with True it is the explicit "recompute from boundary"
        comparison in which old tails are not carried forward.
        """
        boundary = float(boundary)
        if boundary < 0.0:
            raise ValueError("boundary must be non-negative")
        if boundary > 0.0:
            self._render_span(0.0, boundary, block_size)
        if reset_reverbs:
            self.reset_reverb_buses()
        result = self._render_span(boundary, float(duration), block_size)
        self._clock = boundary + float(duration)
        return result

    def get_state(self):
        """JSON-friendly snapshot of all carried state (for checkpoints)."""
        return {
            "clock": self._clock,
            "emitters": {
                e.name: {
                    "playhead": rt.playhead,
                    "env_state": rt.env_state,
                    "gain_l_state": rt.gain_l_state,
                    "gain_r_state": rt.gain_r_state,
                    "lp_state": rt.lp_state,
                    "occl_state": rt.occl_state,
                    "finished": rt.finished,
                    "direct_band_states": rt.direct_band_states,
                    "send_band_states": rt.send_band_states,
                }
                for e in self.emitters
                for rt in [self._runtimes[id(e)]]
            },
            "reverbs": [rv.get_state() for rv in self._reverbs],
            "reverb_gain_states": list(self._reverb_gain_states),
        }

    def set_state(self, state):
        """Restore a snapshot from get_state; rendering continues seamlessly."""
        by_name = {e.name: e for e in self.emitters}
        for name, es in state.get("emitters", {}).items():
            emitter = by_name.get(name)
            if emitter is None:
                continue
            rt = self._runtimes[id(emitter)]
            rt.playhead = float(es["playhead"])
            rt.env_state = float(es["env_state"])
            rt.gain_l_state = float(es["gain_l_state"])
            rt.gain_r_state = float(es["gain_r_state"])
            rt.lp_state = float(es["lp_state"])
            rt.occl_state = float(es["occl_state"])
            rt.finished = bool(es["finished"])
            dbs = es.get("direct_band_states")
            rt.direct_band_states = (
                None if dbs is None else [float(v) for v in dbs])
            sbs = es.get("send_band_states")
            rt.send_band_states = (
                None if sbs is None else [float(v) for v in sbs])
        for rv, rv_state in zip(self._reverbs, state.get("reverbs", [])):
            rv.set_state(rv_state)
        gains = state.get("reverb_gain_states")
        if gains is not None:
            self._reverb_gain_states = [float(g) for g in gains]
        self._clock = float(state.get("clock", 0.0))
