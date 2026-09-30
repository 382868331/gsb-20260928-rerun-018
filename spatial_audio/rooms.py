"""Rooms, doors (portals) and per-room reverbs for occlusion-aware rendering.

Geometry / propagation model:

- Rooms are axis-aligned boxes. A point is "inside" at most one room
  (first match wins); points outside every room form one shared "outside".
- A Portal connects two rooms and has a DoorMotion: keyframed open
  fraction in [0, 1], smoothstep-interpolated so volume/spectrum changes
  are continuous in time and independent of the render block size.
- Sound propagates along the graph of rooms joined by portals. Every
  *simple* path from the source room to the listener room contributes at
  once (a short direct chain and a chain detouring through other rooms
  can therefore arrive together). Each door on a chain multiplies the
  signal by its current coupling, so doors moving anywhere along the
  chain change the result continuously. Switching which paths exist
  (e.g. the listener crossing a boundary) never erases another room's
  reverb tail: tail processing state always lives with the source room
  and only the smooth output gain changes.

Reverbs are feedback comb + allpass filters implemented with scipy
lfilter and carried zi state, so reverb tails are sample-exact regardless
of how the render is split into blocks; the zi arrays can be exported and
restored to suspend a render mid-way and resume it later.
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
    """Rooms + portals; answers multi-path transmission queries.

    Each portal crossing has two independent couplings:

    - dry coupling ``closed + (1 - closed) * openness`` carries the
      spectrally flat (direct) signal;
    - muffled coupling ``(1 - openness)`` is fed by a fixed lowpassed copy
      of the signal, so the heard spectrum moves continuously as the door
      swings.

    For every simple room-to-room path the path gain is the product of
    its couplings; the gains of all paths are summed in parallel. The
    muffled mix of the combined path is the muffled fraction of that
    gain. Chains longer than one portal also keep a small per-hop
    muffling so propagation through several rooms sounds progressively
    duller.
    """

    # Direct sound through a closed portal keeps only a quiet, muffled
    # leakage; an open portal is transparent.
    DIRECT_CLOSED_GAIN = 0.08
    # Reverb tails couple through a closed portal more strongly, so a
    # neighbour's ringing stays distinguishable after the direct sound
    # is blocked.
    REVERB_CLOSED_GAIN = 0.3
    # Extra high-frequency loss per crossed portal on multi-hop chains.
    # Extra muffle fraction accumulated per crossed portal on chains.
    CHAIN_MUFFLE_PER_HOP = 0.12
    # Graph search guard: room scenes are tiny; longer simple paths are
    # acoustically negligible (products of < 1 couplings decay quickly).
    MAX_PATH_HOPS = 5

    def __init__(self, rooms, portals=()):
        self.rooms = list(rooms)
        self.portals = list(portals)
        self._index = {r.name: i for i, r in enumerate(self.rooms)}
        self._simple_paths = {}

    def _enumerate_simple_paths(self, src, dst):
        """All simple paths src -> dst as tuples of room indices.

        Cached per (src, dst); parallel portals between the same two
        rooms are merged into one adjacency edge (their individual
        openness is summed at evaluation time).
        """
        key = (int(src), int(dst))
        cached = self._simple_paths.get(key)
        if cached is not None:
            return cached
        n = len(self.rooms)
        adj = [set() for _ in range(n)]
        for portal in self.portals:
            ia = self._index.get(portal.room_a)
            ib = self._index.get(portal.room_b)
            if ia is None or ib is None:
                continue
            adj[ia].add(ib)
            adj[ib].add(ia)
        paths = []
        if 0 <= src < n and 0 <= dst < n:
            stack = [(src, (src,))]
            while stack:
                node, path = stack.pop()
                if len(path) - 1 >= self.MAX_PATH_HOPS:
                    continue
                for nxt in sorted(adj[node], reverse=True):
                    if nxt in path:
                        continue
                    npath = path + (nxt,)
                    if nxt == dst:
                        paths.append(npath)
                    else:
                        stack.append((nxt, npath))
        self._simple_paths[key] = paths
        return paths

    def _portal_couplings(self, times):
        """Per-portal (dry, muffle) coupling arrays at the given times.

        Returns a dict keyed by unordered room-index pair; when several
        portals join the same room pair their couplings add (parallel
        doors / openings), clipped to a transparent maximum of 1.
        """
        n = len(times)
        edges = {}
        for portal in self.portals:
            ia = self._index.get(portal.room_a)
            ib = self._index.get(portal.room_b)
            if ia is None or ib is None:
                continue
            op = portal.motion.open_fraction(times)
            gain = np.clip(
                self.DIRECT_CLOSED_GAIN
                + (1.0 - self.DIRECT_CLOSED_GAIN) * op,
                0.0, 1.0,
            )
            dry = np.clip(gain * op, 0.0, 1.0)
            wet = np.clip(gain * (1.0 - op), 0.0, 1.0)
            key = (min(ia, ib), max(ia, ib))
            if key in edges:
                prev_dry, prev_wet = edges[key]
                edges[key] = (np.clip(prev_dry + dry, 0.0, 1.0),
                              np.clip(prev_wet + wet, 0.0, 1.0))
            else:
                edges[key] = (dry, wet)
        return edges

    def _reverb_couplings(self, times):
        """Like :meth:`_portal_couplings` for the reverb bus (scalar gain)."""
        edges = {}
        for portal in self.portals:
            ia = self._index.get(portal.room_a)
            ib = self._index.get(portal.room_b)
            if ia is None or ib is None:
                continue
            op = portal.motion.open_fraction(times)
            g = np.clip(
                self.REVERB_CLOSED_GAIN
                + (1.0 - self.REVERB_CLOSED_GAIN) * op,
                0.0, 1.0,
            )
            key = (min(ia, ib), max(ia, ib))
            edges[key] = np.clip(edges.get(key, 0.0) + g, 0.0, 1.0)
        return edges

    def _combined_paths(self, src, dst, times, edge_couplings):
        """Sum (dry_gain, muffled_gain) over all simple src -> dst paths."""
        n = len(times)
        dry_total = np.zeros(n)
        wet_total = np.zeros(n)
        for path in self._enumerate_simple_paths(src, dst):
            dry_p = np.ones(n)
            wet_p = np.zeros(n)
            hops = len(path) - 1
            for a, b in zip(path[:-1], path[1:]):
                dry_e, wet_e = edge_couplings[(min(a, b), max(a, b))]
                new_dry = dry_p * dry_e
                new_wet = wet_p * dry_e + dry_p * wet_e
                dry_p, wet_p = new_dry, new_wet
            if hops > 1:
                chain = min(1.0, self.CHAIN_MUFFLE_PER_HOP * hops)
                wet_p = np.clip(wet_p + dry_p * chain, 0.0, 1.0)
                dry_p = dry_p * (1.0 - chain)
            dry_total += dry_p
            wet_total += wet_p
        return np.clip(dry_total, 0.0, 1.0), np.clip(wet_total, 0.0, 1.0)

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

    def direct_transmission(self, listener_idx, source_idx, times):
        """Per-sample (gain, muffle_mix) summed over all room paths.

        Same room -> (1, 0). Otherwise every simple room-graph path from
        source room to listener room contributes; ``muffle_mix`` is the
        muffled fraction of the summed gain and crossfades to a fixed
        lowpassed copy of the signal. With no connecting path the sound
        is fully blocked (gain 0) rather than artificially leaking.
        """
        n = len(times)
        gain = np.zeros(n)
        mix = np.ones(n)
        edges = self._portal_couplings(times)
        for s0 in np.unique(source_idx):
            for l0 in np.unique(listener_idx):
                s, l = int(s0), int(l0)
                m = (source_idx == s) & (listener_idx == l)
                if not m.any():
                    continue
                if s == l and s >= 0 and l >= 0:
                    gain[m] = 1.0
                    mix[m] = 0.0
                    continue
                if s < 0 or l < 0:
                    continue
                dry, wet = self._combined_paths(s, l, times, edges)
                total = dry + wet
                gain[m] = np.minimum(total, 1.0)[m]
                mix[m] = np.where(
                    total > 1e-12, wet / np.maximum(total, 1e-12), 1.0
                )[m]
        return gain, mix

    def reverb_transmission(self, listener_idx, room_i, times):
        """Per-sample gain at which room_i's reverb reaches the listener.

        Same room -> 1; otherwise all simple room-graph paths contribute.
        """
        n = len(times)
        gain = np.zeros(n)
        edges = self._reverb_couplings(times)
        for l in np.unique(listener_idx):
            m = listener_idx == l
            if not m.any():
                continue
            if l == room_i:
                gain[m] = 1.0
            elif l < 0 or room_i < 0:
                continue
            else:
                g = np.zeros(n)
                for path in self._enumerate_simple_paths(int(room_i), int(l)):
                    gp = np.ones(n)
                    for a, b in zip(path[:-1], path[1:]):
                        gp *= edges[(min(a, b), max(a, b))]
                    g += gp
                gain[m] = np.clip(g, 0.0, 1.0)[m]
        return gain

    def with_door_motions(self, motions):
        """Return a copy whose portals use new DoorMotion values.

        ``motions`` maps portal name to :class:`DoorMotion`; unspecified
        portals keep their motion. Used to replay one recorded take with
        different door states while the source start/stop schedule stays
        exactly the same.
        """
        portals = [
            Portal(p.name, p.room_a, p.room_b,
                   motions.get(p.name, p.motion))
            for p in self.portals
        ]
        return RoomScene(
            [Room.from_dict(r.to_dict()) for r in self.rooms], portals
        )

    def to_dict(self):
        return {
            "rooms": [r.to_dict() for r in self.rooms],
            "portals": [p.to_dict() for p in self.portals],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            [Room.from_dict(r) for r in d.get("rooms", [])],
            [Portal.from_dict(p) for p in d.get("portals", [])],
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
        """Export all filter delay lines for suspend/resume."""
        state = []
        for combs, allpasses in self._channels:
            state.append({
                "combs": [np.asarray(zi).tolist() for _, _, zi in combs],
                "allpasses": [np.asarray(zi).tolist() for _, _, zi in allpasses],
            })
        return state

    def set_state(self, state):
        """Restore delay lines previously produced by :meth:`get_state`."""
        for ch, chs in enumerate(state):
            combs, allpasses = self._channels[ch]
            for i, zi in enumerate(chs["combs"]):
                b, a, _ = combs[i]
                zi = np.asarray(zi, dtype=np.float64)
                if zi.shape != combs[i][2].shape:
                    raise ValueError("comb state length mismatch")
                combs[i] = (b, a, zi.copy())
            for i, zi in enumerate(chs["allpasses"]):
                b, a, _ = allpasses[i]
                zi = np.asarray(zi, dtype=np.float64)
                if zi.shape != allpasses[i][2].shape:
                    raise ValueError("allpass state length mismatch")
                allpasses[i] = (b, a, zi.copy())
