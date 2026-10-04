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


def _coerce_bands(bands):
    """None -> None; otherwise one DoorMotion per band entry."""
    if bands is None:
        return None
    return [
        DoorMotion([(0.0, float(entry))]) if np.isscalar(entry)
        else DoorMotion(entry)
        for entry in bands
    ]


class PanelMotion:
    """Keyframed panel pose: sorted (time, center, yaw) keys.

    Center (xyz) and yaw (rotation about the z axis, radians) are
    smoothstep-interpolated between keys and held constant outside the
    key range, so the pose is a continuous function of time.
    """

    def __init__(self, keys):
        keys = sorted(
            (float(t), tuple(float(v) for v in center), float(yaw))
            for t, center, yaw in keys
        )
        if not keys:
            raise ValueError("PanelMotion needs at least one key")
        self._times = np.array([k[0] for k in keys], dtype=np.float64)
        self._centers = np.array([k[1] for k in keys], dtype=np.float64)
        self._yaws = np.array([k[2] for k in keys], dtype=np.float64)

    def pose_at(self, times):
        """(centers (n, 3), yaws (n,)) at scalar/array time."""
        scalar = np.isscalar(times)
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        times_k, centers, yaws = self._times, self._centers, self._yaws
        if len(times_k) == 1:
            c = np.broadcast_to(centers[0], (len(tv), 3)).copy()
            y = np.full(len(tv), yaws[0])
            return (c[0], float(y[0])) if scalar else (c, y)
        idx = np.clip(np.searchsorted(times_k, tv, side="right") - 1,
                      0, len(times_k) - 2)
        t0, t1 = times_k[idx], times_k[idx + 1]
        u = np.clip((tv - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
        s = (u * u * (3.0 - 2.0 * u))[:, None]
        c = centers[idx] * (1.0 - s) + centers[idx + 1] * s
        y = yaws[idx] * (1.0 - s[:, 0]) + yaws[idx + 1] * s[:, 0]
        before = tv <= times_k[0]
        after = tv >= times_k[-1]
        c[before] = centers[0]
        c[after] = centers[-1]
        y[before] = yaws[0]
        y[after] = yaws[-1]
        return (c[0], float(y[0])) if scalar else (c, y)

    def to_list(self):
        return [
            [float(t), list(c), float(y)]
            for t, c, y in zip(self._times, self._centers, self._yaws)
        ]

    @classmethod
    def from_list(cls, rows):
        return cls([(r[0], r[1], r[2]) for r in rows])


def _coerce_motion(motion):
    if motion is None or isinstance(motion, PanelMotion):
        return motion
    return PanelMotion(motion)


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

    Moving panels: pass motion=PanelMotion (or its key list
    [(t, center, yaw), ...]) to slide/rotate the panel over time. The
    box keeps its size; center/yaw come from the keyframes. Whether a
    propagation path crosses the panel is then evaluated per sample at
    the panel's pose at that moment.

    Revisions: zone.revise(t0, t1, ...) records a new version of the
    panel (bands/gain/muffle/motion) that applies only to propagation
    times inside [t0, t1); outside the interval the base version is
    used. Because band weighting is applied where sound enters the
    reverb bus, tails emitted before the revision keep the historical
    material; only newly generated sound uses the revised panel.
    """

    def __init__(self, name, min_corner, max_corner, gain=0.4, muffle=0.3,
                 bands=None, motion=None):
        self.name = str(name)
        self.min = np.asarray(min_corner, dtype=np.float64)
        self.max = np.asarray(max_corner, dtype=np.float64)
        if self.min.shape != (3,) or self.max.shape != (3,) or np.any(self.min >= self.max):
            raise ValueError("zone corners must be length-3 with min < max")
        self.gain = float(gain)
        self.muffle = float(np.clip(muffle, 0.0, 1.0))
        self.bands = _coerce_bands(bands)
        self.motion = _coerce_motion(motion)
        self._center0 = 0.5 * (self.min + self.max)
        self._half = 0.5 * (self.max - self.min)
        self._revisions = []

    def revise(self, t0, t1, *, label=None, bands=None, gain=None,
               muffle=None, motion=None):
        """Record a revised panel version for propagation times [t0, t1).

        Only the given aspects change; the rest is inherited from the
        base version. Sound generated inside the interval uses the
        revised panel; tails already decaying keep the version they were
        produced with. To recompute from the boundary instead (tails
        regenerated with the new version), record the same revision with
        t0=0.0 so history is re-rendered under the revised panel.
        """
        rev = {
            "t0": float(t0),
            "t1": float(t1),
            "label": str(label) if label is not None
                     else f"rev{len(self._revisions) + 1}",
            "bands": _coerce_bands(bands),
            "gain": None if gain is None else float(gain),
            "muffle": None if muffle is None else float(np.clip(muffle, 0.0, 1.0)),
            "motion": _coerce_motion(motion),
        }
        self._revisions.append(rev)
        return rev

    @property
    def revisions(self):
        return list(self._revisions)

    @property
    def has_bands(self):
        """True when the base version or any revision defines band gains."""
        return self.bands is not None or any(
            r["bands"] is not None for r in self._revisions
        )

    @property
    def has_motion(self):
        return self.motion is not None or any(
            r["motion"] is not None for r in self._revisions
        )

    def version_label_at(self, t):
        """Version name in effect at propagation time t (later wins)."""
        label = "base"
        for rev in self._revisions:
            if rev["t0"] <= t < rev["t1"]:
                label = rev["label"]
        return label

    def version_segments(self, t0, t1):
        """[(start, end, version_label), ...] covering [t0, t1]."""
        bounds = {float(t0), float(t1)}
        for rev in self._revisions:
            for b in (rev["t0"], rev["t1"]):
                if t0 < b < t1:
                    bounds.add(b)
        bounds = sorted(bounds)
        segments = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            segments.append((a, b, self.version_label_at(0.5 * (a + b))))
        return segments

    def pose_at(self, times):
        """(centers, yaws) of the panel at scalar/array time."""
        scalar = np.isscalar(times)
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if self.motion is None:
            centers = np.broadcast_to(self._center0, (len(tv), 3)).copy()
            yaws = np.zeros(len(tv))
        else:
            centers, yaws = self.motion.pose_at(tv)
            centers = np.array(centers, copy=True)
            yaws = np.array(yaws, copy=True)
        for rev in self._revisions:
            if rev["motion"] is None:
                continue
            m = (tv >= rev["t0"]) & (tv < rev["t1"])
            if m.any():
                c, y = rev["motion"].pose_at(tv[m])
                centers[m] = c
                yaws[m] = y
        return (centers[0], float(yaws[0])) if scalar else (centers, yaws)

    def band_gains_at(self, times):
        """Per-band gain at scalar/array time -> (n_bands,) or (n, n_bands).

        Revision-aware: samples inside a revision interval with its own
        bands use the revised material. Samples where the zone is
        broadband (no base bands, no covering revision) get 1.0 so the
        scalar gain/muffle path can handle them. Returns None only when
        no version of the zone ever defines bands.
        """
        if not self.has_bands:
            return None
        scalar = np.isscalar(times)
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if self.bands is not None:
            out = np.stack([m.open_fraction(tv) for m in self.bands], axis=1)
        else:
            n_bands = len(next(r["bands"] for r in self._revisions
                               if r["bands"] is not None))
            out = np.ones((len(tv), n_bands))
        for rev in self._revisions:
            if rev["bands"] is None:
                continue
            m = (tv >= rev["t0"]) & (tv < rev["t1"])
            if m.any():
                out[m] = np.stack(
                    [dm.open_fraction(tv[m]) for dm in rev["bands"]], axis=1
                )
        return out[0] if scalar else out

    def scalar_factors_at(self, times):
        """Per-sample (gain, muffle) for broadband treatment.

        Base gain/muffle apply unless a revision overrides them; samples
        inside a revision that defines per-band gains are neutral here
        (the band weighting replaces the scalar effect for them).
        """
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        gain = np.full(len(tv), self.gain)
        muffle = np.full(len(tv), self.muffle)
        for rev in self._revisions:
            m = (tv >= rev["t0"]) & (tv < rev["t1"])
            if not m.any():
                continue
            if rev["bands"] is not None:
                gain[m] = 1.0
                muffle[m] = 0.0
            if rev["gain"] is not None:
                gain[m] = rev["gain"]
            if rev["muffle"] is not None:
                muffle[m] = rev["muffle"]
        return gain, muffle

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

    def crosses(self, a, b, times=None):
        """Like segments_cross, but evaluated at the panel pose at times.

        times is a scalar or (n,) array matching the segments. For a
        panel without any motion this reduces to segments_cross.
        """
        if times is None or not self.has_motion:
            return self.segments_cross(a, b)
        a, b = np.broadcast_arrays(
            np.atleast_2d(np.asarray(a, dtype=np.float64)),
            np.atleast_2d(np.asarray(b, dtype=np.float64)),
        )
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        n = max(a.shape[0], tv.shape[0])
        a = np.broadcast_to(a, (n, 3))
        b = np.broadcast_to(b, (n, 3))
        tv = np.broadcast_to(tv, (n,))
        centers, yaws = self.pose_at(tv)
        cos, sin = np.cos(yaws), np.sin(yaws)

        def to_local(p):
            d = p - centers
            return np.stack([
                cos * d[:, 0] + sin * d[:, 1],
                -sin * d[:, 0] + cos * d[:, 1],
                d[:, 2],
            ], axis=1)

        la, lb = to_local(a), to_local(b)
        d = lb - la
        t_enter = np.full(len(la), -np.inf)
        t_exit = np.full(len(la), np.inf)
        for axis in range(3):
            da = d[:, axis]
            lo_h, hi_h = -self._half[axis], self._half[axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                t0 = (lo_h - la[:, axis]) / da
                t1 = (hi_h - la[:, axis]) / da
            lo = np.minimum(t0, t1)
            hi = np.maximum(t0, t1)
            still = np.abs(da) < 1e-12
            lo = np.where(still, np.where(la[:, axis] >= lo_h, -np.inf, np.inf), lo)
            hi = np.where(still, np.where(la[:, axis] <= hi_h, np.inf, -np.inf), hi)
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
        if self.motion is not None:
            d["motion"] = self.motion.to_list()
        if self._revisions:
            d["revisions"] = [
                {
                    "t0": r["t0"],
                    "t1": r["t1"],
                    "label": r["label"],
                    "bands": (None if r["bands"] is None
                              else [m.to_list() for m in r["bands"]]),
                    "gain": r["gain"],
                    "muffle": r["muffle"],
                    "motion": (None if r["motion"] is None
                               else r["motion"].to_list()),
                }
                for r in self._revisions
            ]
        return d

    @classmethod
    def from_dict(cls, d):
        zone = cls(d["name"], d["min"], d["max"],
                   d.get("gain", 0.4), d.get("muffle", 0.3),
                   bands=d.get("bands"), motion=d.get("motion"))
        for r in d.get("revisions", []):
            zone.revise(r["t0"], r["t1"], label=r.get("label"),
                        bands=r.get("bands"), gain=r.get("gain"),
                        muffle=r.get("muffle"), motion=r.get("motion"))
        return zone


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

    def __init__(self, rooms, portals=(), zones=(), band_edges=(1000.0,)):
        self.rooms = list(rooms)
        self.portals = list(portals)
        self.zones = list(zones)
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
            for rev in zone.revisions:
                if rev["bands"] is not None and len(rev["bands"]) != self.n_bands:
                    raise ValueError(
                        f"zone {zone.name!r} revision {rev['label']!r} has "
                        f"{len(rev['bands'])} band gains but the scene defines "
                        f"{self.n_bands} bands"
                    )

    @property
    def n_bands(self):
        """Number of frequency bands: len(band_edges) + 1."""
        return len(self.band_edges) + 1

    @property
    def has_band_zones(self):
        """True when any zone carries per-band (frequency-dependent) gains."""
        return any(z.has_bands for z in self.zones)

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

    def _zone_factors(self, waypoints, times=None):
        """Per-sample (gain, muffle) absorbed by zones along a polyline.

        waypoints: list of (n, 3) or (3,) arrays from listener to source
        through door positions. Only segments crossing a zone are damped.
        Zones with per-band gains in their base version are skipped here;
        their effect is applied per frequency band by zone_band_factors
        instead. Broadband zones are evaluated at the panel pose and
        version in effect at each sample's time.
        """
        if not self.zones:
            return None, None
        pts = [np.atleast_2d(np.asarray(w, dtype=np.float64)) for w in waypoints]
        n = max(len(p) for p in pts)
        if times is None:
            times = np.zeros(n)
        times = np.broadcast_to(np.asarray(times, dtype=np.float64), (n,))
        gain = np.ones(n)
        muffle = np.zeros(n)
        for zone in self.zones:
            if zone.bands is not None:
                continue
            hit = np.zeros(n, dtype=bool)
            for a, b in zip(pts[:-1], pts[1:]):
                hit |= zone.crosses(a, b, times)
            zg, zm = zone.scalar_factors_at(times)
            gain = np.where(hit, gain * zg, gain)
            muffle = np.where(hit, 1.0 - (1.0 - muffle) * (1.0 - zm), muffle)
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
            if not zone.has_bands:
                continue
            hit = np.zeros(n, dtype=bool)
            for a, b in zip(pts[:-1], pts[1:]):
                hit |= zone.crosses(a, b, times)
            gains = zone.band_gains_at(times)
            factors = np.where(hit[:, None], factors * gains, factors)
        return factors

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
                zg, zm = self._zone_factors([source_pos, listener_pos], times)
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
                    zg, zm = self._zone_factors(waypoints, times[m])
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
        its own band factor (product of the per-band gains of the zones it
        crosses), weighted by that path's scalar door gain. Returns an
        (n, n_bands) array, or None when no zone has per-band gains.
        """
        if not self.has_band_zones:
            return None
        n = len(times)
        out = np.ones((n, self.n_bands))
        diff = listener_idx != source_idx
        same = ~diff
        if same.any():
            bf = self.zone_band_factors(
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
                bf = self.zone_band_factors(waypoints, times[m])
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
        if not self.has_band_zones:
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
            bf = self.zone_band_factors(
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
                if bool(zone.crosses(a, b, t)[0]):
                    crossed = True
                    break
            if not crossed:
                continue
            entry = {"name": zone.name, "gain": zone.gain,
                     "version": zone.version_label_at(t)}
            bg = zone.band_gains_at(t)
            if bg is not None:
                entry["bands"] = [float(v) for v in bg]
                band_gains *= np.asarray(bg, dtype=np.float64)
                any_band = True
            zones_hit.append(entry)
        return {
            "time": t,
            "listener_room": room_name(li),
            "source_room": room_name(si),
            "portals": [
                {"name": p.name, "open": float(p.motion.open_fraction(t))}
                for p in best
            ],
            "zones": zones_hit,
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
                    zg, _ = self._zone_factors(waypoints, times[m])
                    g_path = g_path * zg
                best = np.maximum(best, g_path)
            gain[m] = np.maximum(best, self.REVERB_NO_PATH_GAIN)
        return gain

    def to_dict(self):
        return {
            "rooms": [r.to_dict() for r in self.rooms],
            "portals": [p.to_dict() for p in self.portals],
            "zones": [z.to_dict() for z in self.zones],
            "band_edges": list(self.band_edges),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            [Room.from_dict(r) for r in d.get("rooms", [])],
            [Portal.from_dict(p) for p in d.get("portals", [])],
            [AbsorptionZone.from_dict(z) for z in d.get("zones", [])],
            band_edges=d.get("band_edges", (1000.0,)),
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
