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

    @property
    def center(self):
        return 0.5 * (self.min + self.max)

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


class Portal:
    """A door connecting two rooms; openness varies with DoorMotion."""

    def __init__(self, name, room_a, room_b, motion=None, position=None):
        self.name = str(name)
        self.room_a = str(room_a)
        self.room_b = str(room_b)
        self.motion = motion if motion is not None else DoorMotion()
        self.position = None if position is None else np.asarray(position, dtype=np.float64)

    def connects(self, a, b):
        return {a, b} == {self.room_a, self.room_b}

    def to_dict(self):
        return {
            "name": self.name,
            "rooms": [self.room_a, self.room_b],
            "motion": self.motion.to_list(),
            "position": None if self.position is None else self.position.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["rooms"][0], d["rooms"][1],
                   DoorMotion.from_list(d.get("motion", [[0.0, 1.0]])),
                   position=d.get("position"))


class AbsorptionZone:
    """Axis-aligned sound-absorbing region (e.g. a treated wall area).

    A zone only affects propagation whose geometric path crosses it: the
    path gain is multiplied by gain and the path picks up extra muffling.
    Paths that do not cross the zone are exactly as loud as without the
    treatment, so the effect stays local instead of making every room
    uniformly quieter.
    """

    def __init__(self, name, min_corner, max_corner, gain=0.4, muffle=0.3):
        self.name = str(name)
        self.min = np.asarray(min_corner, dtype=np.float64)
        self.max = np.asarray(max_corner, dtype=np.float64)
        if self.min.shape != (3,) or self.max.shape != (3,) or np.any(self.min >= self.max):
            raise ValueError("zone corners must be length-3 with min < max")
        self.gain = float(gain)
        self.muffle = float(np.clip(muffle, 0.0, 1.0))

    def segments_cross(self, a, b):
        """Boolean (n,) mask: does segment a[i] -> b[i] intersect this zone?

        a and b are (n, 3) arrays (broadcastable). Slab test per segment.
        """
        a, b = np.broadcast_arrays(
            np.atleast_2d(np.asarray(a, dtype=np.float64)),
            np.atleast_2d(np.asarray(b, dtype=np.float64)),
        )
        d = b - a
        t_enter = np.full(a.shape[0], -np.inf)
        t_exit = np.full(a.shape[0], np.inf)
        for axis in range(3):
            da = d[:, axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                t0 = (self.min[axis] - a[:, axis]) / da
                t1 = (self.max[axis] - a[:, axis]) / da
            lo = np.minimum(t0, t1)
            hi = np.maximum(t0, t1)
            still = np.abs(da) < 1e-12
            # Parallel to the slab: inside -> no constraint, outside -> miss.
            lo = np.where(still, np.where(a[:, axis] >= self.min[axis], -np.inf, np.inf), lo)
            hi = np.where(still, np.where(a[:, axis] <= self.max[axis], np.inf, -np.inf), hi)
            t_enter = np.maximum(t_enter, lo)
            t_exit = np.minimum(t_exit, hi)
        return (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter <= 1.0)

    def to_dict(self):
        return {
            "name": self.name,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "gain": self.gain,
            "muffle": self.muffle,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["min"], d["max"],
                   d.get("gain", 0.4), d.get("muffle", 0.3))


class RoomScene:
    """Rooms + portals; answers direct/reverb transmission queries."""

    # Direct path through a portal: closed door keeps only a quiet,
    # muffled leakage; open door is transparent.
    DIRECT_CLOSED_GAIN = 0.08
    DIRECT_NO_PATH_GAIN = 0.02
    # Reverb heard across rooms: a closed door still lets the other
    # room's tail through clearly enough to be distinguishable.
    REVERB_CLOSED_GAIN = 0.3
    REVERB_NO_PATH_GAIN = 0.05

    def __init__(self, rooms, portals=(), zones=()):
        self.rooms = list(rooms)
        self.portals = list(portals)
        self.zones = list(zones)
        self._index = {r.name: i for i, r in enumerate(self.rooms)}

    def _portal_position(self, portal):
        """Door position: explicit, else midpoint of the two room centers."""
        if portal.position is not None:
            return portal.position
        a = self.rooms[self._index[portal.room_a]].center
        b = self.rooms[self._index[portal.room_b]].center
        return 0.5 * (a + b)

    def _simple_paths(self, idx_a, idx_b, max_paths=8):
        """All simple portal-graph paths between two room indices.

        Each path is a list of portals. With one hop it is the direct
        route; longer paths are detours around closed/missing doors.
        """
        adj = {i: [] for i in range(len(self.rooms))}
        for portal in self.portals:
            ia = self._index.get(portal.room_a)
            ib = self._index.get(portal.room_b)
            if ia is None or ib is None:
                continue
            adj[ia].append((ib, portal))
            adj[ib].append((ia, portal))
        paths = []

        def dfs(node, visited, acc):
            if len(paths) >= max_paths:
                return
            if node == idx_b:
                paths.append(list(acc))
                return
            for nxt, portal in adj.get(node, ()): 
                if nxt in visited:
                    continue
                visited.add(nxt)
                acc.append(portal)
                dfs(nxt, visited, acc)
                acc.pop()
                visited.discard(nxt)

        if idx_a in adj and idx_b in adj:
            dfs(idx_a, {idx_a}, [])
        paths.sort(key=len)
        return paths

    def _zone_factors(self, waypoints):
        """Per-sample (gain, muffle) absorbed by zones along a polyline.

        waypoints: list of (n, 3) or (3,) arrays from listener to source
        through door positions. Only segments crossing a zone are damped.
        """
        if not self.zones:
            return None, None
        pts = [np.atleast_2d(np.asarray(w, dtype=np.float64)) for w in waypoints]
        n = max(len(p) for p in pts)
        gain = np.ones(n)
        muffle = np.zeros(n)
        for zone in self.zones:
            hit = np.zeros(n, dtype=bool)
            for a, b in zip(pts[:-1], pts[1:]):
                hit |= zone.segments_cross(a, b)
            gain = np.where(hit, gain * zone.gain, gain)
            muffle = np.where(hit, 1.0 - (1.0 - muffle) * (1.0 - zone.muffle), muffle)
        return gain, muffle

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
        """Per-sample (gain, muffle_mix) for the direct + detour paths.

        Every simple portal-graph path between the listener room and the
        source room contributes: each path's gain is the product of its
        doors' transmission, and path gains add (capped at 1). A closed
        door on the direct route therefore still lets a detour through
        other rooms be heard, distinctly from having no path at all.

        muffle_mix in [0, 1] crossfades to a fixed lowpassed copy of the
        signal, so the spectrum changes continuously with door motion.
        Absorption zones only damp paths whose geometry crosses them.
        """
        n = len(times)
        gain = np.ones(n)
        mix = np.zeros(n)
        diff = listener_idx != source_idx
        if not diff.any():
            if self.zones and listener_pos is not None and source_pos is not None:
                zg, zm = self._zone_factors([source_pos, listener_pos])
                gain = gain * zg
                mix = zm
            return gain, mix

        gain[diff] = self.DIRECT_NO_PATH_GAIN
        mix[diff] = 1.0
        pairs = set(zip(listener_idx[diff].tolist(), source_idx[diff].tolist()))
        for li, si in pairs:
            m = diff & (listener_idx == li) & (source_idx == si)
            if li < 0 or si < 0:
                continue
            paths = self._simple_paths(li, si)
            if not paths:
                continue
            g_sum = np.zeros(m.sum())
            g_weighted_mix = np.zeros(m.sum())
            for path in paths:
                open_prod = np.ones(m.sum())
                for portal in path:
                    open_prod = open_prod * portal.motion.open_fraction(times[m])
                g_path = self.DIRECT_CLOSED_GAIN ** len(path) + (
                    (1.0 - self.DIRECT_CLOSED_GAIN) ** len(path)
                ) * open_prod
                mix_path = 1.0 - open_prod
                if self.zones and listener_pos is not None and source_pos is not None:
                    waypoints = [listener_pos[m]] + [
                        self._portal_position(portal) for portal in path
                    ] + [source_pos[m]]
                    zg, zm = self._zone_factors(waypoints)
                    g_path = g_path * zg
                    mix_path = 1.0 - (1.0 - mix_path) * (1.0 - zm)
                g_sum += g_path
                g_weighted_mix += g_path * mix_path
            total = np.minimum(g_sum, 1.0)
            gain[m] = np.maximum(total, self.DIRECT_NO_PATH_GAIN)
            mix[m] = np.where(
                g_sum > 1e-12, g_weighted_mix / np.maximum(g_sum, 1e-12), 1.0
            )
        return gain, mix

    def reverb_transmission(self, listener_idx, room_i, times, listener_pos=None):
        """Per-sample gain at which room_i's reverb reaches the listener.

        Multi-hop rooms contribute along their strongest portal path, so a
        far room's tail is still audible (fainter) through open doors.
        """
        n = len(times)
        gain = np.full(n, self.REVERB_NO_PATH_GAIN)
        same = listener_idx == room_i
        gain[same] = 1.0
        rest = ~same
        if not rest.any():
            return gain
        if room_i < 0:
            return gain
        for li in set(listener_idx[rest].tolist()):
            m = rest & (listener_idx == li)
            if li < 0:
                continue
            paths = self._simple_paths(li, room_i)
            if not paths:
                continue
            best = np.zeros(m.sum())
            for path in paths:
                open_prod = np.ones(m.sum())
                for portal in path:
                    open_prod = open_prod * portal.motion.open_fraction(times[m])
                g_path = self.REVERB_CLOSED_GAIN ** len(path) + (
                    (1.0 - self.REVERB_CLOSED_GAIN) ** len(path)
                ) * open_prod
                if self.zones and listener_pos is not None:
                    waypoints = [listener_pos[m]] + [
                        self._portal_position(portal) for portal in path
                    ] + [self.rooms[room_i].center]
                    zg, _ = self._zone_factors(waypoints)
                    g_path = g_path * zg
                best = np.maximum(best, g_path)
            gain[m] = np.maximum(best, self.REVERB_NO_PATH_GAIN)
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
        """JSON-friendly snapshot of every carried filter state."""
        return [
            {
                "combs": [zi.tolist() for _, _, zi in combs],
                "allpasses": [zi.tolist() for _, _, zi in allpasses],
            }
            for combs, allpasses in self._channels
        ]

    def set_state(self, state):
        """Restore a snapshot from get_state (tail keeps decaying)."""
        for ch_state, (combs, allpasses) in zip(state, self._channels):
            for i, zi in enumerate(ch_state["combs"]):
                b, a, _ = combs[i]
                combs[i] = (b, a, np.asarray(zi, dtype=np.float64))
            for i, zi in enumerate(ch_state["allpasses"]):
                b, a, _ = allpasses[i]
                allpasses[i] = (b, a, np.asarray(zi, dtype=np.float64))
