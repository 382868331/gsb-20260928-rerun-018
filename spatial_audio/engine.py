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
from .rooms import Door, Occluder, Room, room_weights, segment_intersections

SPEED_OF_SOUND = 343.0
_NYQUIST_CUTOFF = 20000.0


def _geo_scan(drive, rho, seed, offset=0):
    """Solve y[k] = drive[k] + rho*y[k-1] with given y[-1] = seed.

    Vectorized in 128-sample windows anchored to the global index
    ``offset`` of drive[0]; scalar fragments handle the ends. Window
    boundaries therefore never depend on processing block size.
    """
    m = len(drive)
    y = np.empty(m)
    prev = float(seed)
    pos = 0
    while pos < m:
        to_next = 128 - ((offset + pos) % 128)
        # A fragment that is the *start* of a chain can begin exactly on a
        # boundary (to_next == 128); only mid-chain fragments are scalar.
        end = min(pos + to_next, m)
        if not (to_next == 128 and end - pos == 128):
            for i in range(pos, end):
                prev = prev * float(rho[i]) + float(drive[i])
                y[i] = prev
            pos = end
            continue
        r = rho[pos:end]
        # General: y[i] = P[i]*prev + sum_{j<=i} P[i]/S[j]*drive[j], where
        # P[i] = prod_{0..i} r, S[j] = prod_{0..j} r (drive at j carries
        # prod_{j+1..i} r = P[i]/S[j]).
        rr = np.maximum(r, 1e-300)
        seed_w = np.exp(np.cumsum(np.log(rr)))           # P[i]
        drv = drive[pos:end]
        through = seed_w                                  # S[j] = P[j]
        weighted = np.cumsum(drv / np.maximum(through, 1e-300))
        y[pos:end] = seed_w * prev + seed_w * weighted
        prev = float(y[end - 1])
        pos = end
    return y


def _varying_one_pole(signal, cutoff, sample_rate, prev, sample_offset):
    """Per-sample low-pass with a smooth cutoff trajectory; block invariant.

    Vectorized via cumulative products; ``state`` is the previous output and
    is carried across blocks, so the result never depends on block size.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x, prev
    nyq = 0.5 * sample_rate
    alpha = 1.0 - np.exp(-2.0 * np.pi * np.minimum(cutoff, nyq * 0.999) / sample_rate)
    alpha = np.clip(alpha, 1e-7, 1.0)
    q = 1.0 - alpha
    # Closed-form scan inside globally-aligned 128-sample windows; short head
    # and tail fragments use scalar recursion. Anchoring windows to the
    # absolute sample grid keeps the output identical for any block size.
    y = np.empty(n)
    pos = 0
    while pos < n:
        to_next = 128 - ((sample_offset + pos) % 128)
        end = min(pos + to_next, n)
        if to_next == 128 and end - pos == 128:
            sl = slice(pos, end)
            y[sl] = _geo_scan(
                alpha[sl] * x[sl], q[sl], prev,
                offset=sample_offset + pos,
            )
            prev = float(y[end - 1])
        else:
            for i in range(pos, end):
                prev = q[i] * prev + alpha[i] * x[i]
                y[i] = prev
        pos = end
    return y, prev


class _RoomReverb:
    """Cheap feedback-comb reverb with persistent state.

    Four parallel comb filters per channel with a damping one-pole inside
    the loop, summed into the room wet signal. States are carried block to
    block, so the natural tail survives across block boundaries and even
    after sources stop; rendering a longer duration only extends the tail.
    """

    DELAYS_L = (1559, 1613, 1721, 1777)
    DELAYS_R = (1571, 1627, 1699, 1789)

    def __init__(self, index, room, sample_rate):
        self.index = index
        self.room = room
        self.sample_rate = sample_rate
        mean_delay = 0.125 * (sum(self.DELAYS_L) + sum(self.DELAYS_R))
        feedback = 10.0 ** (-3.0 * mean_delay / (max(room.rt60, 0.05) * sample_rate))
        self.feedback = float(np.clip(feedback, 0.0, 0.985))
        self.damping = float(
            1.0 - np.exp(-2.0 * np.pi * 3500.0 / sample_rate)
        )
        # Per comb: D-length ring buffers for input x, comb-stage u and final
        # output y, indexed by global sample position, plus the cursor. This
        # makes short and long processing blocks share identical state.
        self.state = {
            ch: [(np.zeros(d), np.zeros(d), np.zeros(d), 0)
                 for d in (self.DELAYS_L, self.DELAYS_R)[ch]]
            for ch in range(2)
        }

    def process(self, sends):
        """Inject per-sample sends and return the wet contribution.

        Damped feedback comb, factored into two stages:

        * u[n] = x[n-D] + g*u[n-D]  (pure feedback comb; splits into
          independent residue-class chains),
        * y[n] = damp*u[n] + (1-damp)*y[n-1]  (damping integrator).

        Both stages carry their state in D-length rings indexed by the
        global sample position, which is what makes every block split
        produce identical results.
        """
        n = len(sends)
        wet = np.zeros((n, 2))
        damp = self.damping
        q = 1.0 - damp
        g = self.feedback
        for ch in range(2):
            for ci, delay in enumerate((self.DELAYS_L, self.DELAYS_R)[ch]):
                x = sends[:, ch]
                xhist, uhist, yhist, cursor = self.state[ch][ci]
                if n < delay:
                    # Short block: all delayed indices are distinct and live
                    # in the rings; vectorize the lookup, no length-D arrays.
                    gi = np.arange(cursor, cursor + n)
                    delayed_slots = (gi - delay) % delay
                    active = gi >= delay
                    x_delayed = np.where(active, xhist[delayed_slots], 0.0)
                    u_delayed = np.where(active, uhist[delayed_slots], 0.0)
                    u = x_delayed + g * u_delayed
                else:
                    # Build [-D, n) windows from the rings.
                    gidx = np.arange(cursor - delay, cursor + n)
                    slots = gidx % delay
                    valid = gidx >= 0
                    xglob = np.where(valid, xhist[slots], 0.0)
                    xglob[delay:] = x
                    uglob = np.where(valid, uhist[slots], 0.0)
                    u = uglob[delay:]
                    chain_count = cursor // delay
                    first_residue = int(cursor % delay)
                    for local_r in range(delay):
                        idx = np.arange(local_r, n, delay)
                        delayed = xglob[idx]
                        first_global = cursor + local_r
                        seed_slot = first_global % delay
                        seed = float(uhist[seed_slot])
                        gr = int(first_global % delay)
                        step_offset = chain_count + (
                            1 if gr >= first_residue else 0
                        )
                        u[idx] = _geo_scan(
                            delayed, np.full(idx.size, g), seed,
                            offset=step_offset,
                        )
                # Stage 2: damping integrator y[n] = damp*u[n] + q*y[n-1].
                prev_y = (float(yhist[(cursor - 1) % delay]) if cursor > 0
                          else 0.0)
                out = lfilter(
                    [damp], [1.0, -q], u, zi=[q * prev_y]
                )[0]
                wet[:, ch] += 0.25 * out
                new_slots = (np.arange(cursor, cursor + n)) % delay
                xhist[new_slots] = x
                uhist[new_slots] = u
                yhist[new_slots] = out
                self.state[ch][ci] = (xhist, uhist, yhist, cursor + n)
        return wet


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
        self.occ_gain_state = 1.0  # smoothed occlusion linear gain
        self.lp_state = 0.0        # per-emitter occlusion low-pass output
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
        rooms=None,
        occluders=None,
        occlusion_smooth_ms=4.0,
        door_leak=0.08,
        reverb_return_ms=40.0,
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
        self.rooms = list(rooms or [])
        self.occluders = list(occluders or [])
        self._oc_smooth_coef = float(
            1.0 - np.exp(-dt / (occlusion_smooth_ms / 1000.0))
        )
        self.door_leak = float(door_leak)
        self.reverb_return_ms = float(reverb_return_ms)
        self._reverbs = [
            _RoomReverb(i, room, self.sample_rate) for i, room in enumerate(self.rooms)
        ]
        self._adjacency = self._build_adjacency()
        self._sample_cursor = 0
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

    def add_room(self, room):
        self.rooms.append(room)
        self._reverbs.append(_RoomReverb(len(self.rooms) - 1, room, self.sample_rate))
        self._adjacency = self._build_adjacency()
        return room

    def add_occluder(self, occluder):
        self.occluders.append(occluder)
        self._adjacency = self._build_adjacency()
        return occluder

    def _build_adjacency(self):
        """Map (room_i, room_j) -> connecting door, for coupled reverb."""
        adj = {}
        if len(self.rooms) < 2:
            return adj
        for wall in self.occluders:
            if not wall.is_door():
                continue
            mid = 0.5 * (wall.a + wall.b)
            hits = [i for i, r in enumerate(self.rooms) if r.contains(mid)]
            if len(hits) >= 2:
                i, j = sorted(hits[:2])
                adj[(i, j)] = wall
        return adj

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
        sample_offset = self._sample_cursor
        self._sample_cursor += n
        times = time_seconds + (np.arange(n) + 0.5) / self.sample_rate

        lpos, lvel = self._listener_pose(times)
        fwd = self.listener.forward
        right = self.listener.right

        mixed = np.zeros((n, 2), dtype=np.float64)
        peaks = {}
        sb = [self._smooth_coef]
        sa = [1.0, -(1.0 - self._smooth_coef)]
        nr = len(self.rooms)
        room_sends = np.zeros((n, nr)) if nr else None
        lw = room_weights(lpos, self.rooms) if nr else None
        door_open = {
            id(w): w.openness_at(times) for w in self.occluders if w.is_door()
        }

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

            gate = self._gate(emitter, rt, time_seconds, times)

            occ_gain, occ_cutoff = self._occlusion(
                epos, lpos, times, door_open, rt, time_seconds
            )
            src_lp, rt.lp_state = _varying_one_pole(
                src, occ_cutoff, self.sample_rate, rt.lp_state, sample_offset
            )

            dist_gain = 1.0 / (1.0 + dist / self.reference_distance)

            # Equal-power pan from angle to the listener's right axis.
            x = np.clip(np.sum(rhat * right, axis=1), -1.0, 1.0)
            gl = np.sqrt(np.clip(0.5 * (1.0 - x), 0.0, 1.0))
            gr = np.sqrt(np.clip(0.5 * (1.0 + x), 0.0, 1.0))
            # Mild front/back cue.
            frontness = np.clip(np.sum(rhat * fwd, axis=1), 0.0, 1.0)
            rear_gain = 0.85 + 0.15 * frontness

            gain_l = dist_gain * gl * rear_gain * emitter.volume * occ_gain
            gain_r = dist_gain * gr * rear_gain * emitter.volume * occ_gain

            gl_s, zi = lfilter(sb, sa, gain_l, zi=[rt.gain_l_state])
            rt.gain_l_state = float(zi[0])
            gr_s, zi = lfilter(sb, sa, gain_r, zi=[rt.gain_r_state])
            rt.gain_r_state = float(zi[0])

            ch_l = src_lp * gl_s * gate
            ch_r = src_lp * gr_s * gate
            mixed[:, 0] += ch_l
            mixed[:, 1] += ch_r
            peaks[emitter.name] = max(
                peaks.get(emitter.name, 0.0),
                float(np.max(np.abs(ch_l))),
                float(np.max(np.abs(ch_r))),
            )

            if nr:
                # The dry source excites its own room; sound also leaks into
                # neighbouring rooms through connecting doors.
                ew = room_weights(epos, self.rooms)
                send = src * gate * emitter.volume
                room_sends += ew * send[:, None]
                for (ri, rj), door in self._adjacency.items():
                    op = door_open[id(door)]
                    # Transmission through a connecting door: a constant
                    # leak plus the open fraction. Both factors multiply
                    # the *other* room's (smooth) weight, so a source walking
                    # through the doorway changes excitation continuously.
                    link = self.door_leak + (1.0 - self.door_leak) * op
                    room_sends[:, ri] += ew[:, rj] * send * link
                    room_sends[:, rj] += ew[:, ri] * send * link

        if nr:
            for ri, rev in enumerate(self._reverbs):
                s = room_sends[:, ri]
                wet = rev.process(np.column_stack((s, s)))
                level = self.rooms[ri].reverb_level
                mixed += (lw[:, ri] * level)[:, None] * wet

        return mixed, peaks

    def _occlusion(self, epos, lpos, times, door_open, rt, block_start):
        """Smooth per-sample occlusion gain and low-pass cutoff.

        Every wall crossing is treated as diffraction around the nearer
        wall endpoint: the excess path length is zero for a ray grazing the
        endpoint, so entering/exiting a doorway changes level and spectrum
        continuously. Door blocking scales with (1 - openness).
        """
        n = len(times)
        gain_db = np.zeros(n)
        cutoff = np.full(n, _NYQUIST_CUTOFF)
        p0 = epos[:, :2]
        p1 = lpos[:, :2]
        straight = np.linalg.norm(p1 - p0, axis=1)
        for wall in self.occluders:
            crossed, u, v = segment_intersections(p0, p1, wall.a, wall.b)
            if not np.any(crossed):
                continue
            near = (v < 0.5)[:, None]
            endpoint = np.where(near, wall.a, wall.b)
            detour = (
                np.linalg.norm(p0 - endpoint, axis=1)
                + np.linalg.norm(p1 - endpoint, axis=1)
                - straight
            )
            detour = np.maximum(detour, 0.0)
            k = np.clip(detour / wall.diffraction_scale, 0.0, 1.0)
            if wall.is_door():
                op = door_open[id(wall)]
                blocked = 1.0 - op
                # Closed-door band eases in over a short detour so a ray
                # grazing the door frame is unaffected (continuity).
                seal = 0.5 - 0.5 * np.cos(
                    np.pi * np.clip(detour / 0.15, 0.0, 1.0)
                )
                strength = blocked
                w_db = -(
                    wall.max_attenuation_db * k
                    + wall.closed_attenuation_db * seal
                ) * strength
                ratio = (wall.min_cutoff_hz / _NYQUIST_CUTOFF) ** (
                    k * strength
                ) * (wall.closed_cutoff_hz / _NYQUIST_CUTOFF) ** (
                    seal * strength
                )
            else:
                w_db = -wall.max_attenuation_db * k
                ratio = (wall.min_cutoff_hz / _NYQUIST_CUTOFF) ** k
            gain_db = np.where(crossed, gain_db + w_db, gain_db)
            cutoff = np.where(crossed, np.minimum(cutoff, _NYQUIST_CUTOFF * ratio), cutoff)

        target_gain = 10.0 ** (gain_db / 20.0)
        coef = self._oc_smooth_coef
        smoothed, zi = lfilter(
            [coef], [1.0, -(1.0 - coef)], target_gain, zi=[rt.occ_gain_state]
        )
        rt.occ_gain_state = float(zi[0])
        return smoothed, np.clip(cutoff, 20.0, _NYQUIST_CUTOFF)

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
