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

    Frequency-band treatment: pass bands with one entry per scene band
    (see RoomScene.band_edges). Each entry is either a constant gain or a
    keyframe list [(time, gain), ...] interpolated like a DoorMotion, so a
    material can be swapped mid-replay. A zone with bands applies those
    per-band gains instead of the scalar gain/muffle on the direct path
    and on the reverb send, letting one material kill high frequencies
    while leaving lows almost untouched.
    """

    def __init__(self, name, min_corner, max_corner, gain=0.4, muffle=0.3,
                 bands=None):
        self.name = str(name)
        self.min = np.asarray(min_corner, dtype=np.float64)
        self.max = np.asarray(max_corner, dtype=np.float64)
        if self.min.shape != (3,) or self.max.shape != (3,) or np.any(self.min >= self.max):
            raise ValueError("zone corners must be length-3 with min < max")
        self.gain = float(gain)
        self.muffle = float(np.clip(muffle, 0.0, 1.0))
        if bands is None:
            self.bands = None
        else:
            self.bands = [
                DoorMotion([(0.0, float(entry))]) if np.isscalar(entry)
                else DoorMotion(entry)
                for entry in bands
            ]

    def band_gains_at(self, times):
        """Per-band gain at scalar/array time -> (n_bands,) or (n, n_bands).

        Returns None for a purely broadband (scalar) zone.
        """
        if self.bands is None:
            return None
        scalar = np.isscalar(times)
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        out = np.stack([m.open_fraction(tv) for m in self.bands], axis=1)
        return out[0] if scalar else out

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
        d = {
            "name": self.name,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "gain": self.gain,
            "muffle": self.muffle,
        }
        if self.bands is not None:
            d["bands"] = [m.to_list() for m in self.bands]
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["min"], d["max"],
                   d.get("gain", 0.4), d.get("muffle", 0.3),
                   bands=d.get("bands"))


def _smooth_curve(times, values, query):
    tv = np.atleast_1d(np.asarray(query, dtype=np.float64))
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times) == 1:
        return np.broadcast_to(values[0], (len(tv), values.shape[1])).copy()
    idx = np.clip(np.searchsorted(times, tv, side="right") - 1, 0, len(times) - 2)
    t0, t1 = times[idx], times[idx + 1]
    v0, v1 = values[idx], values[idx + 1]
    u = np.clip((tv - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
    s = (u * u * (3.0 - 2.0 * u))[:, None]
    out = v0 + (v1 - v0) * s
    out[tv <= times[0]] = values[0]
    out[tv >= times[-1]] = values[-1]
    return out


class PanelRevision:
    """A panel edit authoritative only on one explicit time interval.

    Positions/headings are keyed in global scene time and smoothstep
    interpolated. A band entry of None keeps the base panel transmission
    for that band; a scalar applies constantly, while keyframed rows are
    interpolated. Revisions do not touch sound already held in a reverb
    bus: such a tail keeps the material present when it was produced.
    """

    def __init__(self, name, start, end, position_keys=None,
                 heading_keys=None, bands=None):
        self.name = str(name)
        self.start = float(start)
        self.end = float(end)
        if not self.start < self.end:
            raise ValueError("revision needs start < end")
        self.position_keys = position_keys
        self.heading_keys = heading_keys
        self.bands = list(bands) if bands is not None else None

    def sample_positions(self, times):
        if self.position_keys is None:
            return None
        rows = sorted((float(t), np.asarray(p, dtype=np.float64))
                      for t, p in self.position_keys)
        kt = np.array([t for t, _ in rows])
        kv = np.stack([p for _, p in rows])
        return _smooth_curve(kt, kv, times)

    def sample_headings(self, times):
        if self.heading_keys is None:
            return None
        rows = sorted((float(t), float(h)) for t, h in self.heading_keys)
        kt = np.array([t for t, _ in rows])
        kv = np.unwrap(np.array([h for _, h in rows]))[:, None]
        return _smooth_curve(kt, kv, times)[:, 0]

    def sample_bands(self, n_bands, times):
        if self.bands is None:
            return None
        if len(self.bands) != n_bands:
            raise ValueError(
                f"revision {self.name!r} has {len(self.bands)} band entries "
                f"but the panel uses {n_bands} bands"
            )
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        out = np.full((len(tv), n_bands), np.nan)
        for bi, entry in enumerate(self.bands):
            if entry is None:
                continue
            if np.isscalar(entry):
                out[:, bi] = float(entry)
            else:
                rows = sorted((float(t), float(g)) for t, g in entry)
                kt = np.array([t for t, _ in rows])
                kv = np.array([[g] for _, g in rows])
                out[:, bi] = _smooth_curve(kt, kv, tv)[:, 0]
        return out

    def to_dict(self):
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "position_keys": (
                None if self.position_keys is None
                else [[float(t), *np.asarray(p, dtype=np.float64).tolist()]
                      for t, p in self.position_keys]
            ),
            "heading_keys": (
                None if self.heading_keys is None
                else [[float(t), float(h)] for t, h in self.heading_keys]
            ),
            "bands": self.bands,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d.get("name", "revision"), d["start"], d["end"],
            position_keys=(
                None if d.get("position_keys") is None
                else [(r[0], r[1:4]) for r in d["position_keys"]]
            ),
            heading_keys=(
                None if d.get("heading_keys") is None
                else [(r[0], r[1]) for r in d["heading_keys"]]
            ),
            bands=d.get("bands"),
        )


class Panel:
    """Finite oriented band-transmission panel with editable placement.

    width spans the horizontal in-panel axis and height spans z. heading
    is the normal heading about z (0 points +x, pi/2 points +y). Position
    and heading keyframes are smoothstep interpolated, so the panel can
    slide or rotate during a take. bands gives one gain per RoomScene
    frequency band; each is constant or DoorMotion keyframed. Revisions
    override placement and/or gains only on their own intervals.
    """

    def __init__(self, name, width, height,
                 position_keys=((0.0, (0.0, 0.0, 0.0)),),
                 heading_keys=((0.0, 0.0),), bands=None, revisions=()):
        self.name = str(name)
        self.width = float(width)
        self.height = float(height)
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("panel width and height must be positive")
        rows = sorted((float(t), np.asarray(p, dtype=np.float64))
                      for t, p in position_keys)
        self._pos_times = np.array([t for t, _ in rows])
        self._pos_values = np.stack([p for _, p in rows])
        hrows = sorted((float(t), float(h)) for t, h in heading_keys)
        self._head_times = np.array([t for t, _ in hrows])
        self._head_values = np.unwrap(np.array([h for _, h in hrows]))
        self.bands = None if bands is None else [
            DoorMotion([(0.0, float(entry))]) if np.isscalar(entry)
            else DoorMotion(entry)
            for entry in bands
        ]
        self.revisions = list(revisions)

    def base_pose_at(self, times):
        """Base (position (n,3), heading (n,)), ignoring interval edits."""
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        pos = _smooth_curve(self._pos_times, self._pos_values, tv)
        head = _smooth_curve(self._head_times, self._head_values[:, None], tv)[:, 0]
        return pos, head

    def base_band_gains_at(self, times):
        if self.bands is None:
            return None
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        return np.stack([m.open_fraction(tv) for m in self.bands], axis=1)

    def version_at(self, times):
        """Active revision name per sample ('base' outside every edit)."""
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        names = np.full(len(tv), "base", dtype=object)
        for revision in self.revisions:
            names[(tv >= revision.start) & (tv < revision.end)] = revision.name
        return names

    def pose_at(self, times):
        """Effective (positions, headings) after interval revisions."""
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        pos, head = self.base_pose_at(tv)
        for revision in self.revisions:
            mask = (tv >= revision.start) & (tv < revision.end)
            if not mask.any():
                continue
            rp = revision.sample_positions(tv[mask])
            if rp is not None:
                pos[mask] = rp
            rh = revision.sample_headings(tv[mask])
            if rh is not None:
                head[mask] = rh
        return pos, head

    def band_gains_at(self, times):
        """Effective per-band transmission as an (n, n_bands) array."""
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        gains = self.base_band_gains_at(tv)
        for revision in self.revisions:
            mask = (tv >= revision.start) & (tv < revision.end)
            if not mask.any() or gains is None:
                continue
            rg = revision.sample_bands(len(self.bands), tv[mask])
            if rg is None:
                continue
            gains[mask] = np.where(np.isnan(rg), gains[mask], rg)
        return gains

    def frames_at(self, times):
        """Per-sample (center, normal, width-axis) geometry frames."""
        centers, headings = self.pose_at(times)
        normals = np.stack(
            [np.cos(headings), np.sin(headings), np.zeros_like(headings)], axis=1
        )
        width_axes = np.stack(
            [-np.sin(headings), np.cos(headings), np.zeros_like(headings)], axis=1
        )
        return centers, normals, width_axes

    def segments_cross(self, a, b, times):
        """Does segment a[i] -> b[i] cross the finite panel at time t[i]?"""
        a = np.atleast_2d(np.asarray(a, dtype=np.float64))
        b = np.atleast_2d(np.asarray(b, dtype=np.float64))
        centers, normals, width_axes = self.frames_at(times)
        delta = b - a
        den = np.sum(delta * normals, axis=1)
        side = np.sum((a - centers) * normals, axis=1)
        hit = np.abs(den) > 1e-12
        alpha = np.zeros_like(den)
        alpha[hit] = -side[hit] / den[hit]
        hit &= (alpha >= 0.0) & (alpha <= 1.0)
        q = a + alpha[:, None] * delta
        rel = q - centers
        hit &= np.abs(np.sum(rel * width_axes, axis=1)) <= 0.5 * self.width
        hit &= np.abs(rel[:, 2]) <= 0.5 * self.height
        return hit

    def to_dict(self):
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "position_keys": [
                [float(t), *self._pos_values[i].tolist()]
                for i, t in enumerate(self._pos_times)
            ],
            "heading_keys": [
                [float(t), float(h)]
                for t, h in zip(self._head_times, self._head_values)
            ],
            "bands": None if self.bands is None
                    else [m.to_list() for m in self.bands],
            "revisions": [r.to_dict() for r in self.revisions],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["name"], d["width"], d["height"],
            position_keys=[
                (r[0], r[1:4])
                for r in d.get("position_keys", [[0.0, [0.0, 0.0, 0.0]]])
            ],
            heading_keys=[
                (r[0], r[1]) for r in d.get("heading_keys", [[0.0, 0.0]])
            ],
            bands=d.get("bands"),
            revisions=[PanelRevision.from_dict(r)
                        for r in d.get("revisions", [])],
        )


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

    def __init__(self, rooms, portals=(), zones=(), band_edges=(1000.0,),
                 panels=()):
        self.rooms = list(rooms)
        self.portals = list(portals)
        self.zones = list(zones)
        self.panels = list(panels)
        self._index = {r.name: i for i, r in enumerate(self.rooms)}
        edges = tuple(float(e) for e in band_edges)
        if any(e <= 0.0 for e in edges) or tuple(sorted(edges)) != edges:
            raise ValueError("band_edges must be positive and sorted")
        self.band_edges = edges
        for zone in self.zones:
            if zone.bands is not None and len(zone.bands) != self.n_bands:
                raise ValueError(
                    f"zone {zone.name!r} has {len(zone.bands)} band gains but "
                    f"the scene defines {self.n_bands} bands"
                )
        for panel in self.panels:
            if panel.bands is not None and len(panel.bands) != self.n_bands:
                raise ValueError(
                    f"panel {panel.name!r} has {len(panel.bands)} band gains but "
                    f"the scene defines {self.n_bands} bands"
                )

    @property
    def n_bands(self):
        """Number of frequency bands: len(band_edges) + 1."""
        return len(self.band_edges) + 1

    @property
    def has_band_zones(self):
        """True when any zone carries per-band (frequency-dependent) gains."""
        return any(z.bands is not None for z in self.zones)

    @property
    def has_band_processing(self):
        """True when zones or oriented panels define per-band gains."""
        return self.has_band_zones or any(p.bands is not None for p in self.panels)

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
        Zones with per-band gains are skipped here; their effect is
        applied per frequency band by zone_band_factors instead.
        """
        if not self.zones:
            return None, None
        pts = [np.atleast_2d(np.asarray(w, dtype=np.float64)) for w in waypoints]
        n = max(len(p) for p in pts)
        gain = np.ones(n)
        muffle = np.zeros(n)
        for zone in self.zones:
            if zone.bands is not None:
                continue
            hit = np.zeros(n, dtype=bool)
            for a, b in zip(pts[:-1], pts[1:]):
                hit |= zone.segments_cross(a, b)
            gain = np.where(hit, gain * zone.gain, gain)
            muffle = np.where(hit, 1.0 - (1.0 - muffle) * (1.0 - zone.muffle), muffle)
        return gain, muffle

    def zone_band_factors(self, waypoints, times):
        """Per-sample, per-band gain absorbed by band zones along a polyline.

        Returns an (n, n_bands) array, or None when no zone defines
        per-band gains. Segments crossing several band zones multiply
        their per-band gains, so passing two panels differs from one.
        """
        if not self.has_band_zones:
            return None
        pts = [np.atleast_2d(np.asarray(w, dtype=np.float64)) for w in waypoints]
        n = max(len(p) for p in pts)
        times = np.broadcast_to(np.asarray(times, dtype=np.float64), (n,))
        factors = np.ones((n, self.n_bands))
        for zone in self.zones:
            if zone.bands is None:
                continue
            hit = np.zeros(n, dtype=bool)
            for a, b in zip(pts[:-1], pts[1:]):
                hit |= zone.segments_cross(a, b)
            gains = zone.band_gains_at(times)
            factors = np.where(hit[:, None], factors * gains, factors)
        return factors

    def panel_band_factors(self, waypoints, times):
        """Per-sample, per-band gain from oriented panels on a polyline.

        Every panel the listener-to-source polyline crosses at that same
        listen time multiplies its band transmission. Geometry and gains
        are both evaluated per sample, so moving/rotating a panel changes
        only newly rendered sound; reverb bus state is never rewritten.
        Returns (n, n_bands) or None.
        """
        if not self.panels or not any(p.bands is not None for p in self.panels):
            return None
        pts = [np.atleast_2d(np.asarray(w, dtype=np.float64)) for w in waypoints]
        n = max(len(p) for p in pts)
        tv = np.broadcast_to(np.asarray(times, dtype=np.float64), (n,))
        factors = np.ones((n, self.n_bands))
        for panel in self.panels:
            if panel.bands is None:
                continue
            hit = np.zeros(n, dtype=bool)
            for a, b in zip(pts[:-1], pts[1:]):
                hit |= panel.segments_cross(a, b, tv)
            gains = panel.band_gains_at(tv)
            factors = np.where(hit[:, None], factors * gains, factors)
        return factors

    def combined_band_factors(self, waypoints, times):
        """Per-band factors of both axis-aligned zones and panels."""
        factors = np.ones((len(np.atleast_1d(times)), self.n_bands))
        zone_factors = self.zone_band_factors(waypoints, times)
        if zone_factors is not None:
            factors *= zone_factors
        panel_factors = self.panel_band_factors(waypoints, times)
        if panel_factors is not None:
            factors *= panel_factors
        return factors if self.has_band_processing else None

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
        Zones/panels only damp paths whose geometry crosses them.
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

    def direct_zone_bands(self, listener_idx, source_idx, times,
                          listener_pos, source_pos):
        """Per-sample, per-band extra gain from band zones on direct paths.

        Mirrors direct_transmission: every portal-graph path contributes
        its own band factor (product of the per-band gains of the zones and
        panels it crosses), weighted by that path's scalar door gain.
        Returns an (n, n_bands) array, or None without band materials.
        """
        if not self.has_band_processing:
            return None
        n = len(times)
        out = np.ones((n, self.n_bands))
        diff = listener_idx != source_idx
        same = ~diff
        if same.any():
            bf = self.combined_band_factors(
                [source_pos[same], listener_pos[same]], times[same]
            )
            out[same] = bf
        if not diff.any():
            return out
        pairs = set(zip(listener_idx[diff].tolist(), source_idx[diff].tolist()))
        for li, si in pairs:
            m = diff & (listener_idx == li) & (source_idx == si)
            if li < 0 or si < 0:
                continue
            paths = self._simple_paths(li, si)
            if not paths:
                continue
            g_sum = np.zeros(m.sum())
            b_sum = np.zeros((m.sum(), self.n_bands))
            for path in paths:
                open_prod = np.ones(m.sum())
                for portal in path:
                    open_prod = open_prod * portal.motion.open_fraction(times[m])
                g_path = self.DIRECT_CLOSED_GAIN ** len(path) + (
                    (1.0 - self.DIRECT_CLOSED_GAIN) ** len(path)
                ) * open_prod
                waypoints = [listener_pos[m]] + [
                    self._portal_position(portal) for portal in path
                ] + [source_pos[m]]
                bf = self.combined_band_factors(waypoints, times[m])
                g_sum += g_path
                b_sum += g_path[:, None] * bf
            out[m] = np.where(
                g_sum[:, None] > 1e-12,
                b_sum / np.maximum(g_sum, 1e-12)[:, None],
                1.0,
            )
        return out

    def send_band_factors(self, source_pos, source_idx, times):
        """Per-sample, per-band zone gain for the reverb send of a source.

        Applied where the sound enters its room's reverb bus, so a tail
        already decaying keeps the spectrum of the materials that were in
        place when it was produced; swapping a material later only colors
        newly generated sound. Returns (n, n_bands) or None.
        """
        if not self.has_band_processing:
            return None
        n = len(times)
        out = np.ones((n, self.n_bands))
        centers = np.zeros((n, 3))
        valid = np.zeros(n, dtype=bool)
        for ri, room in enumerate(self.rooms):
            m = source_idx == ri
            if m.any():
                centers[m] = room.center
                valid[m] = True
        if valid.any():
            bf = self.combined_band_factors(
                [source_pos[valid], centers[valid]], times[valid]
            )
            out[valid] = bf
        return out

    def explain_path(self, listener_pos, source_pos, time):
        """Dominant propagation path at one listen moment, with materials.

        Returns a dict naming the strongest portal route between the
        listener and the source at the given time, each door's open
        fraction, every absorption zone the route crosses (with its
        per-band gains at that moment) and the resulting scalar and
        per-band path gains. Lets a user trace why a moment sounds the
        way it does.
        """
        lp = np.asarray(listener_pos, dtype=np.float64)
        sp = np.asarray(source_pos, dtype=np.float64)
        t = float(time)
        li = int(self.room_index_at([lp])[0])
        si = int(self.room_index_at([sp])[0])
        room_name = lambda i: self.rooms[i].name if 0 <= i < len(self.rooms) else None
        candidates = []
        if li == si:
            candidates.append((1.0, []))
        elif li >= 0 and si >= 0:
            for path in self._simple_paths(li, si):
                open_prod = 1.0
                for portal in path:
                    open_prod *= portal.motion.open_fraction(t)
                g = self.DIRECT_CLOSED_GAIN ** len(path) + (
                    (1.0 - self.DIRECT_CLOSED_GAIN) ** len(path)
                ) * open_prod
                candidates.append((g, path))
        score, best = max(candidates, key=lambda c: c[0]) if candidates else (0.0, [])
        waypoints = [lp] + [self._portal_position(p) for p in best] + [sp]
        zones_hit = []
        band_gains = np.ones(self.n_bands)
        any_band = False
        for zone in self.zones:
            crossed = False
            for a, b in zip(waypoints[:-1], waypoints[1:]):
                if bool(zone.segments_cross(a, b)[0]):
                    crossed = True
                    break
            if not crossed:
                continue
            entry = {"name": zone.name, "gain": zone.gain}
            bg = zone.band_gains_at(t)
            if bg is not None:
                entry["bands"] = [float(v) for v in bg]
                band_gains *= np.asarray(bg, dtype=np.float64)
                any_band = True
            zones_hit.append(entry)
        panels_hit = []
        for panel in self.panels:
            if panel.bands is None:
                continue
            crossed = False
            for a, b in zip(waypoints[:-1], waypoints[1:]):
                if bool(panel.segments_cross(a, b, [t])[0]):
                    crossed = True
                    break
            if not crossed:
                continue
            bg = panel.band_gains_at([t])[0]
            center, heading = panel.pose_at([t])
            panels_hit.append({
                "name": panel.name,
                "version": str(panel.version_at([t])[0]),
                "position": [float(v) for v in center[0]],
                "heading": float(heading[0]),
                "bands": [float(v) for v in bg],
            })
            band_gains *= bg
            any_band = True
        return {
            "time": t,
            "listener_room": room_name(li),
            "source_room": room_name(si),
            "portals": [
                {"name": p.name, "open": float(p.motion.open_fraction(t))}
                for p in best
            ],
            "zones": zones_hit,
            "panels": panels_hit,
            "path_gain": float(score),
            "band_gains": [float(v) for v in band_gains] if any_band else None,
        }

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
            "panels": [p.to_dict() for p in self.panels],
            "band_edges": list(self.band_edges),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            [Room.from_dict(r) for r in d.get("rooms", [])],
            [Portal.from_dict(p) for p in d.get("portals", [])],
            [AbsorptionZone.from_dict(z) for z in d.get("zones", [])],
            band_edges=d.get("band_edges", (1000.0,)),
            panels=[Panel.from_dict(p) for p in d.get("panels", [])],
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

    def reset_state(self):
        """Replace every carried delay line with silence."""
        for ci, (combs, allpasses) in enumerate(self._channels):
            self._channels[ci] = (
                [(b, a, np.zeros_like(zi)) for b, a, zi in combs],
                [(b, a, np.zeros_like(zi)) for b, a, zi in allpasses],
            )
