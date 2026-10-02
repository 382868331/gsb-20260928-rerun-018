"""Rooms, doors (portals) and per-room reverbs for occlusion-aware rendering.

Geometry model:

- Rooms are axis-aligned boxes. A point is "inside" at most one room
  (first match wins); points outside every room form one shared "outside".
- A Portal connects two rooms and has a DoorMotion: keyframed open
  fraction in [0, 1], smoothstep-interpolated so volume/spectrum changes
  are continuous in time and independent of the render block size.
- Direct-path transmission between the listener room and a source room:
  same room -> (gain 1, no muffling); through a portal -> gain and
  lowpass mix follow the open fraction; no connecting portal -> heavy
  occlusion.

Reverbs are built from feedback comb + allpass filters implemented with
scipy lfilter and carried zi state, so reverb tails are sample-exact
regardless of how the render is split into blocks.

Multi-room propagation:

- Rooms and portals form a graph. Direct sound between two rooms travels
  along every simple portal path between them: the short "direct" route
  and longer detours around closed doors. Each path's gain is the product
  of its doors' per-sample transmission, so a closing door continuously
  removes its route while detour routes keep carrying sound.
- AbsorptionZone boxes model added absorptive treatment. A zone only
  attenuates propagation whose geometric path crosses the zone (direct
  listener<->source segment, or listener<->room-centre reverb path); it
  never lowers the level of rooms wholesale. Zones apply pure gain, no
  extra muffling, so absorption stays spectrally distinguishable from a
  closing door (which also lowpasses and cuts reverb transmission).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


class DoorMotion:
    """Keyframed door opening: sorted (time, open_fraction) keys.

    Fractions are smoothstep-interpolated between keys and held constant
    outside the key range, so the opening is a continuous function of time.
    """

    def __init__(self, keys=((0.0, 1.0),)):
        keys = sorted((float(t), float(f)) for t, f in keys)
        if not keys:
            keys = [(0.0, 1.0)]
        self._times = np.array([k[0] for k in keys], dtype=np.float64)
        self._fracs = np.clip(np.array([k[1] for k in keys], dtype=np.float64), 0.0, 1.0)

    def open_fraction(self, t):
        """Open fraction in [0, 1] at scalar or array-like time t."""
        scalar = np.isscalar(t)
        tv = np.atleast_1d(np.asarray(t, dtype=np.float64))
        times, fracs = self._times, self._fracs
        if len(times) == 1:
            out = np.full(len(tv), fracs[0])
            return float(out[0]) if scalar else out
        idx = np.clip(np.searchsorted(times, tv, side="right") - 1, 0, len(times) - 2)
        t0, t1 = times[idx], times[idx + 1]
        f0, f1 = fracs[idx], fracs[idx + 1]
        u = np.clip((tv - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
        s = u * u * (3.0 - 2.0 * u)
        out = f0 + (f1 - f0) * s
        out[tv <= times[0]] = fracs[0]
        out[tv >= times[-1]] = fracs[-1]
        return float(out[0]) if scalar else out

    def to_list(self):
        return [[float(t), float(f)] for t, f in zip(self._times, self._fracs)]

    @classmethod
    def from_list(cls, rows):
        return cls([(r[0], r[1]) for r in rows])


class Room:
    """Axis-aligned room box with its own reverb characteristics."""

    def __init__(self, name, min_corner, max_corner, reverb_time=0.6, reverb_send=0.25):
        self.name = str(name)
        self.min = np.asarray(min_corner, dtype=np.float64)
        self.max = np.asarray(max_corner, dtype=np.float64)
        if self.min.shape != (3,) or self.max.shape != (3,) or np.any(self.min >= self.max):
            raise ValueError("room corners must be length-3 with min < max")
        self.reverb_time = float(reverb_time)
        self.reverb_send = float(reverb_send)

    def mask(self, positions):
        """Boolean mask of points (n, 3) lying inside this room."""
        p = np.asarray(positions, dtype=np.float64)
        return np.all((p >= self.min) & (p <= self.max), axis=1)

    def to_dict(self):
        return {
            "name": self.name,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "reverb_time": self.reverb_time,
            "reverb_send": self.reverb_send,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["min"], d["max"],
                   d.get("reverb_time", 0.6), d.get("reverb_send", 0.25))


def _segments_cross_box(p0, p1, bmin, bmax):
    """Boolean mask: does segment p0[i]->p1[i] intersect the box?

    Slab test, vectorized over n segments. An endpoint inside the box
    counts as crossing.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    tmin = np.zeros(len(p0))
    tmax = np.ones(len(p0))
    for ax in range(3):
        d = p1[:, ax] - p0[:, ax]
        lo, hi = bmin[ax], bmax[ax]
        parallel = np.abs(d) < 1e-12
        safe_d = np.where(parallel, 1.0, d)
        ta = (lo - p0[:, ax]) / safe_d
        tb = (hi - p0[:, ax]) / safe_d
        tlo = np.minimum(ta, tb)
        thi = np.maximum(ta, tb)
        inside = (p0[:, ax] >= lo) & (p0[:, ax] <= hi)
        # Parallel segments either span the whole slab or miss it.
        tlo = np.where(parallel, np.where(inside, -np.inf, np.inf), tlo)
        thi = np.where(parallel, np.where(inside, np.inf, -np.inf), thi)
        tmin = np.maximum(tmin, tlo)
        tmax = np.minimum(tmax, thi)
    return tmax >= tmin


class AbsorptionZone:
    """Axis-aligned box of added absorptive treatment.

    Only propagation paths that geometrically cross the zone are
    attenuated (by 1 - absorption per crossing zone); everything else
    is bit-identical to the untreated scene.
    """

    def __init__(self, name, min_corner, max_corner, absorption=0.7):
        self.name = str(name)
        self.min = np.asarray(min_corner, dtype=np.float64)
        self.max = np.asarray(max_corner, dtype=np.float64)
        if self.min.shape != (3,) or self.max.shape != (3,) or np.any(self.min >= self.max):
            raise ValueError("zone corners must be length-3 with min < max")
        self.absorption = float(np.clip(absorption, 0.0, 1.0))

    @property
    def gain_factor(self):
        return 1.0 - self.absorption

    def to_dict(self):
        return {
            "name": self.name,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "absorption": self.absorption,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["min"], d["max"], d.get("absorption", 0.7))


class Portal:
    """A door connecting two rooms; openness varies with DoorMotion."""

    def __init__(self, name, room_a, room_b, motion=None):
        self.name = str(name)
        self.room_a = str(room_a)
        self.room_b = str(room_b)
        self.motion = motion if motion is not None else DoorMotion()

    def connects(self, a, b):
        return {a, b} == {self.room_a, self.room_b}

    def to_dict(self):
        return {
            "name": self.name,
            "rooms": [self.room_a, self.room_b],
            "motion": self.motion.to_list(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["rooms"][0], d["rooms"][1],
                   DoorMotion.from_list(d.get("motion", [[0.0, 1.0]])))


class RoomScene:
    """Rooms + portals + absorption zones; answers transmission queries.

    Direct and reverb transmission between two rooms follows every simple
    portal path in the room graph, so detour routes around a closed door
    keep contributing while the direct route fades out.
    """

    # Direct path through a portal: closed door keeps only a quiet,
    # muffled leakage; open door is transparent.
    DIRECT_CLOSED_GAIN = 0.08
    DIRECT_NO_PATH_GAIN = 0.02
    # Reverb heard across rooms: a closed door still lets the other
    # room's tail through clearly enough to be distinguishable.
    REVERB_CLOSED_GAIN = 0.3
    REVERB_NO_PATH_GAIN = 0.05

    # Longest detour (in portals) considered for propagation paths.
    MAX_PATH_HOPS = 6

    def __init__(self, rooms, portals=(), zones=()):
        self.rooms = list(rooms)
        self.portals = list(portals)
        self.zones = list(zones)
        self._index = {r.name: i for i, r in enumerate(self.rooms)}
        self._adj = {i: [] for i in range(len(self.rooms))}
        for pi, portal in enumerate(self.portals):
            ia = self._index.get(portal.room_a)
            ib = self._index.get(portal.room_b)
            if ia is None or ib is None:
                continue
            self._adj[ia].append((pi, ib))
            self._adj[ib].append((pi, ia))
        self._path_cache = {}

    def _paths_between(self, a, b):
        """All simple portal paths between room indices a and b.

        Each path is a tuple of portal indices. Results are cached; the
        graph is fixed for the scene's lifetime.
        """
        key = (a, b)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        found = []

        def dfs(node, visited, acc):
            if len(acc) >= self.MAX_PATH_HOPS:
                return
            for pi, nxt in self._adj.get(node, ()):
                if nxt in visited:
                    continue
                acc.append(pi)
                if nxt == b:
                    found.append(tuple(acc))
                else:
                    dfs(nxt, visited | {nxt}, acc)
                acc.pop()

        if a != b and a in self._adj and b in self._adj:
            dfs(a, {a}, [])
        self._path_cache[key] = found
        return found

    def _zone_factor(self, p0, p1):
        """Per-sample gain multiplier from absorption zones crossed by
        the segments p0[i] -> p1[i] (1.0 when no zone is crossed)."""
        if not self.zones:
            return 1.0
        factor = np.ones(len(p0))
        for zone in self.zones:
            cross = _segments_cross_box(p0, p1, zone.min, zone.max)
            factor = np.where(cross, factor * zone.gain_factor, factor)
        return factor

    def room_index_at(self, positions):
        """Room index per point (n, 3); -1 when outside every room."""
        p = np.asarray(positions, dtype=np.float64)
        idx = np.full(len(p), -1, dtype=np.int64)
        unset = np.ones(len(p), dtype=bool)
        for i, room in enumerate(self.rooms):
            m = unset & room.mask(p)
            idx[m] = i
            unset[m] = False
        return idx

    def direct_transmission(self, listener_idx, source_idx, times,
                            listener_pos=None, source_pos=None):
        """Per-sample (gain, muffle_mix) for the direct path.

        muffle_mix in [0, 1] crossfades to a fixed lowpassed copy of the
        signal, so the spectrum changes continuously with door motion.
        Gain sums over every simple portal path (direct route plus
        detours); each path's muffle is gain-weighted into the mix.
        """
        n = len(times)
        gain = np.ones(n)
        mix = np.zeros(n)
        diff = listener_idx != source_idx
        if diff.any():
            gain[diff] = self.DIRECT_NO_PATH_GAIN
            mix[diff] = 1.0
            pairs = set(
                zip(listener_idx[diff].tolist(), source_idx[diff].tolist())
            )
            for li, si in pairs:
                m = diff & (listener_idx == li) & (source_idx == si)
                paths = self._paths_between(li, si) if li >= 0 and si >= 0 else []
                if not paths:
                    continue
                total = np.zeros(int(m.sum()))
                weighted_mix = np.zeros(int(m.sum()))
                for path in paths:
                    pgain = np.ones(int(m.sum()))
                    popen = np.ones(int(m.sum()))
                    for pi in path:
                        op = self.portals[pi].motion.open_fraction(times[m])
                        pgain *= (
                            self.DIRECT_CLOSED_GAIN
                            + (1.0 - self.DIRECT_CLOSED_GAIN) * op
                        )
                        popen *= op
                    total += pgain
                    weighted_mix += pgain * (1.0 - popen)
                gain[m] = total
                mix[m] = weighted_mix / np.maximum(total, 1e-12)
            if self.zones and listener_pos is not None and source_pos is not None:
                factor = self._zone_factor(listener_pos, source_pos)
                gain = gain * factor
        return gain, mix

    def reverb_transmission(self, listener_idx, room_i, times, listener_pos=None):
        """Per-sample gain at which room_i's reverb reaches the listener.

        Multi-hop: the tail propagates along every portal path from
        room_i to the listener's room (summed, capped at 1), so a tail
        keeps decaying through detours when a door on the direct route
        closes.
        """
        n = len(times)
        gain = np.full(n, self.REVERB_NO_PATH_GAIN)
        same = listener_idx == room_i
        gain[same] = 1.0
        rest = ~same
        if rest.any():
            for li in set(listener_idx[rest].tolist()):
                m = rest & (listener_idx == li)
                paths = (
                    self._paths_between(room_i, li)
                    if li >= 0 and room_i >= 0 else []
                )
                if not paths:
                    continue
                total = np.zeros(int(m.sum()))
                for path in paths:
                    pgain = np.ones(int(m.sum()))
                    for pi in path:
                        op = self.portals[pi].motion.open_fraction(times[m])
                        pgain *= (
                            self.REVERB_CLOSED_GAIN
                            + (1.0 - self.REVERB_CLOSED_GAIN) * op
                        )
                    total += pgain
                gain[m] = np.clip(total, 0.0, 1.0)
        if self.zones and listener_pos is not None:
            centre = (self.rooms[room_i].min + self.rooms[room_i].max) / 2.0
            p1 = np.broadcast_to(centre, (n, 3))
            gain = gain * self._zone_factor(listener_pos, p1)
        return gain

    def to_dict(self):
        return {
            "rooms": [r.to_dict() for r in self.rooms],
            "portals": [p.to_dict() for p in self.portals],
            "zones": [z.to_dict() for z in self.zones],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            [Room.from_dict(r) for r in d.get("rooms", [])],
            [Portal.from_dict(p) for p in d.get("portals", [])],
            [AbsorptionZone.from_dict(z) for z in d.get("zones", [])],
        )


class RoomReverb:
    """Stereo feedback-comb reverb with carried filter state.

    All filters run through scipy lfilter with zi saved between blocks,
    so the tail is bit-identical for any block size.
    """

    _COMB_DELAYS = (0.0297, 0.0371, 0.0411, 0.0437)
    _ALLPASS_DELAYS = (0.0050, 0.0017)
    _ALLPASS_GAIN = 0.5

    def __init__(self, sample_rate, reverb_time=0.6, channel_scales=(1.0, 1.13)):
        self._channels = []
        for scale in channel_scales:
            combs = []
            for d in self._COMB_DELAYS:
                delay = max(1, int(round(d * scale * sample_rate)))
                g = 10.0 ** (-3.0 * delay / (sample_rate * max(reverb_time, 1e-3)))
                a = np.zeros(delay + 1)
                a[0] = 1.0
                a[delay] = -g
                combs.append(([1.0], a, np.zeros(delay)))
            allpasses = []
            for d in self._ALLPASS_DELAYS:
                delay = max(1, int(round(d * scale * sample_rate)))
                g = self._ALLPASS_GAIN
                b = np.zeros(delay + 1)
                b[0] = -g
                b[delay] = 1.0
                a = np.zeros(delay + 1)
                a[0] = 1.0
                a[delay] = -g
                allpasses.append((b, a, np.zeros(delay)))
            self._channels.append((combs, allpasses))

    def process(self, mono):
        """mono (n,) -> wet stereo (n, 2); state carries across calls."""
        mono = np.asarray(mono, dtype=np.float64)
        out = np.empty((len(mono), 2))
        for ch, (combs, allpasses) in enumerate(self._channels):
            y = np.zeros(len(mono))
            for i, (b, a, zi) in enumerate(combs):
                c, zf = lfilter(b, a, mono, zi=zi)
                combs[i] = (b, a, zf)
                y += c
            y *= 0.25
            for i, (b, a, zi) in enumerate(allpasses):
                y, zf = lfilter(b, a, y, zi=zi)
                allpasses[i] = (b, a, zf)
            out[:, ch] = y
        return out

    def get_state(self):
        """JSON-serializable carried filter state (the still-decaying tail)."""
        return [
            {
                "combs": [zi.tolist() for _, _, zi in combs],
                "allpasses": [zi.tolist() for _, _, zi in allpasses],
            }
            for combs, allpasses in self._channels
        ]

    def set_state(self, state):
        """Restore a state produced by get_state (tail keeps decaying)."""
        for ch_state, (combs, allpasses) in zip(state, self._channels):
            for i, zi in enumerate(ch_state["combs"]):
                b, a, _ = combs[i]
                combs[i] = (b, a, np.asarray(zi, dtype=np.float64))
            for i, zi in enumerate(ch_state["allpasses"]):
                b, a, _ = allpasses[i]
                allpasses[i] = (b, a, np.asarray(zi, dtype=np.float64))
