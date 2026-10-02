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


# Default frequency band layout shared by scenes/zones/engine. Zones refer
# to bands by name; a zone that only lists some names leaves the other
# bands exactly as loud as without treatment.
DEFAULT_BANDS = ("low", "mid", "high")
DEFAULT_BAND_EDGES = (400.0, 2500.0)  # Hz: low |<e0| mid |<e1| high



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
    """Axis-aligned, frequency-band-aware absorbing region.

    A zone only affects propagation whose geometric path crosses it.
    Material effect is given per frequency band via ``band_gains`` (a
    mapping band name -> linear gain, <= 1): e.g. a porous panel may keep
    the low band nearly intact while strongly cutting the high band.
    Paths that do not cross the zone are bit-identical to the untreated
    scene, so treatment stays local instead of colouring every room.

    The material may be installed / changed / removed while a replay is
    running: ``schedule`` maps a material key to keyframes
    ``[(time, band_gains_or_gain, muffle), ...]``. Gains are linearly
    interpolated per band between keyframes of the same key; switching
    keys crossfades over ``blend`` seconds so the spectrum never clicks.
    Already propagated sound (e.g. a decaying reverb tail sent before a
    material change) keeps the conditions it actually passed through;
    time-varying gains are evaluated per sample on newly propagating
    signal only.
    """

    def __init__(self, name, min_corner, max_corner, gain=0.4, muffle=0.3,
                 band_gains=None, bands=DEFAULT_BANDS,
                 schedule=None, blend=0.05):
        self.name = str(name)
        self.min = np.asarray(min_corner, dtype=np.float64)
        self.max = np.asarray(max_corner, dtype=np.float64)
        if self.min.shape != (3,) or self.max.shape != (3,) or np.any(self.min >= self.max):
            raise ValueError("zone corners must be length-3 with min < max")
        self.bands = tuple(bands)
        self._materials = {}
        # Legacy scalar description, preserved exactly for round trips.
        self._scalar_gain = float(gain) if band_gains is None else None
        self._scalar_muffle = float(np.clip(muffle, 0.0, 1.0)) if band_gains is None else None
        # Default (time-independent) material, kept under key "base".
        self._materials["base"] = self._make_material(
            band_gains if band_gains is not None else gain, muffle
        )
        self.active_key = "base"
        self.blend = float(max(blend, 0.0))
        # key -> list of (time, band_gain array, muffle) keyframes
        self._keyframes = {}
        if schedule:
            for key, frames in schedule.items():
                self.set_schedule(
                    key,
                    [(t, bg, mf) for t, bg, mf in frames],
                )

    def _make_material(self, gain_like, muffle):
        """Normalise a scalar gain or per-band mapping into a gain vector."""
        muffle = float(np.clip(muffle, 0.0, 1.0))
        if np.isscalar(gain_like) or (
            isinstance(gain_like, np.ndarray) and gain_like.ndim == 0
        ):
            g = float(gain_like)
            # Legacy behaviour: broadband gain plus a high-frequency tilt.
            tilt = list(np.linspace(0.0, 1.0, len(self.bands)))
            gains = {
                band: g * (1.0 - muffle * t)
                for band, t in zip(self.bands, tilt)
            }
        else:
            gain_map = dict(gain_like)
            gains = {b: 1.0 for b in self.bands}
            for band, value in gain_map.items():
                if band not in gains:
                    raise ValueError(f"unknown band {band!r} for zone {self.name!r}")
                gains[band] = float(value)
            # Unknown bands to the caller pass unchanged; honour a scalar
            # broadband gain via muffle tilt is impossible for custom
            # bands, so only the explicit mapping applies.
            muffle = 0.0
        vec = np.array(
            [float(np.clip(gains.get(band, 1.0), 0.0, 1.0)) for band in self.bands]
        )
        return {"gains": vec, "muffle": muffle}

    # -- legacy scalar accessors (back-compatible broadband behaviour) ---

    @property
    def gain(self):
        if self._scalar_gain is not None:
            return self._scalar_gain
        return float(self._materials["base"]["gains"].min())

    @property
    def muffle(self):
        if self._scalar_muffle is not None:
            return self._scalar_muffle
        return float(self._materials["base"]["muffle"])

    def band_gain(self, band):
        """Time-independent gain of the base material for one band."""
        i = self.bands.index(band)
        return float(self._materials["base"]["gains"][i])

    # -- time-varying materials -----------------------------------------

    def set_material(self, key, band_gains, muffle=0.0):
        """Register (or replace) a named material arrangement."""
        self._materials[str(key)] = self._make_material(band_gains, muffle)

    def use_material(self, key, at_time=None):
        """Select a registered material arrangement from now on."""
        key = str(key)
        if key not in self._materials:
            raise KeyError(f"unknown material {key!r} for zone {self.name!r}")
        self.active_key = key

    def set_schedule(self, key, frames):
        """Keyframe a material's per-band gains over time.

        frames: [(time, gain_scalar_or_band_mapping, muffle), ...].
        The material is inactive (no absorption) before its first frame.
        """
        key = str(key)
        if key not in self._materials:
            # Register a placeholder; first frame defines the gains.
            self._materials[key] = self._make_material(1.0, 0.0)
        norm = []
        for t, gain_like, mf in frames:
            mat = self._make_material(gain_like, mf)
            norm.append((float(t), mat["gains"], mat["muffle"]))
        norm.sort(key=lambda row: row[0])
        self._keyframes[key] = norm
        self.active_key = key

    def material_at(self, times):
        """Per-sample (band_gains (n, b), muffle (n,)) applied at times."""
        tv = np.atleast_1d(np.asarray(times, dtype=np.float64))
        n, b = len(tv), len(self.bands)
        gains_out = np.ones((n, b))
        muffle_out = np.zeros(n)

        active = self._materials[self.active_key]
        static_g = np.broadcast_to(active["gains"], (n, b)).copy()
        static_m = np.full(n, active["muffle"])

        frames = self._keyframes.get(self.active_key)
        if not frames:
            return static_g, static_m

        # Before the first keyframe the base material applies unchanged;
        # afterwards the keyframed gains take over with a short blend so
        # the spectrum never clicks at the install moment.
        ft = np.array([f[0] for f in frames])
        garr = np.stack([f[1] for f in frames])
        marr = np.array([f[2] for f in frames])
        key_g = np.empty((n, b))
        for j in range(b):
            # Constant extension beyond the last keyframe.
            key_g[:, j] = np.interp(tv, ft, garr[:, j],
                                    left=static_g[0, j], right=garr[-1, j])
        key_m = np.interp(tv, ft, marr, left=static_m[0], right=marr[-1])
        if self.blend > 0.0:
            weight = np.clip((tv - ft[0]) / self.blend, 0.0, 1.0)
        else:
            weight = (tv >= ft[0]).astype(float)
        gains_out = static_g * (1.0 - weight[:, None]) + key_g * weight[:, None]
        muffle_out = static_m * (1.0 - weight) + key_m * weight
        return gains_out, muffle_out

    @property
    def dynamic(self):
        """True when the active material changes while a replay runs."""
        return bool(self._keyframes.get(self.active_key))

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
        schedule = {}
        for key, frames in self._keyframes.items():
            schedule[key] = [
                [t, g.tolist(), m] for t, g, m in frames
            ]
        return {
            "name": self.name,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "bands": list(self.bands),
            "materials": {
                key: {"gains": mat["gains"].tolist(), "muffle": mat["muffle"]}
                for key, mat in self._materials.items()
            },
            "active_key": self.active_key,
            "blend": self.blend,
            "schedule": schedule,
            "gain": self._scalar_gain,
            "muffle": self._scalar_muffle,
        }

    @classmethod
    def from_dict(cls, d):
        bands = tuple(d.get("bands", DEFAULT_BANDS))
        materials = d.get("materials")
        if materials and d.get("gain") is None and d.get("muffle") is None:
            base = materials.get("base", next(iter(materials.values())))
            zone = cls(d["name"], d["min"], d["max"],
                       band_gains=dict(zip(bands, base["gains"])),
                       bands=bands, blend=d.get("blend", 0.05))
            for key, mat in materials.items():
                zone._materials[key] = {
                    "gains": np.asarray(mat["gains"], dtype=np.float64),
                    "muffle": float(mat["muffle"]),
                }
            zone.active_key = d.get("active_key", "base")
        else:
            zone = cls(
                d["name"], d["min"], d["max"],
                gain=d.get("gain", 0.4), muffle=d.get("muffle", 0.3),
                bands=bands, blend=d.get("blend", 0.05),
            )
        for key, frames in d.get("schedule", {}).items():
            zone._keyframes[key] = [
                (float(t), np.asarray(g, dtype=np.float64), float(m))
                for t, g, m in frames
            ]
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

    def __init__(self, rooms, portals=(), zones=(),
                 bands=DEFAULT_BANDS, band_edges=DEFAULT_BAND_EDGES):
        self.rooms = list(rooms)
        self.portals = list(portals)
        self.zones = list(zones)
        self.bands = tuple(bands)
        self.band_edges = tuple(band_edges)
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

    def _zone_factors(self, waypoints, times=None, dynamic_filter=None):
        """Per-sample band gains absorbed by zones along a polyline.

        waypoints: list of (n, 3) or (3,) arrays from listener to source
        through door positions. Only segments crossing a zone are damped;
        crossing several zones multiplies their per-band gains, so passing
        through two treated areas differs from passing through one.
        Material gains are evaluated at the propagation time.
        """
        pts = [np.atleast_2d(np.asarray(w, dtype=np.float64)) for w in waypoints]
        n = len(times) if times is not None else max(len(p) for p in pts)
        times = np.zeros(n) if times is None else np.asarray(times)
        gain = np.ones((n, len(self.bands)))
        muffle = np.zeros(n)
        for zone in self.zones:
            if dynamic_filter is not None and zone.dynamic != dynamic_filter:
                continue
            hit = np.zeros(n, dtype=bool)
            for a, bpt in zip(pts[:-1], pts[1:]):
                hit |= zone.segments_cross(a, bpt)
            if not hit.any():
                continue
            zg, zm = zone.material_at(times)
            hit2 = hit[:, None]
            gain = np.where(hit2, gain * zg, gain)
            muffle = np.where(hit, 1.0 - (1.0 - muffle) * (1.0 - zm), muffle)
        return gain, muffle

    def send_absorption(self, source_idx, times, source_pos):
        """Per-sample band gains colouring the wet send into room reverbs.

        Only *dynamic* (keyframed) materials are applied here, at the
        source -> room-centre segment at emission time. Sound already
        stored in a decaying tail is therefore never re-coloured by a
        material change made later. Static materials continue to damp the
        audible tail on the listener-side route via reverb_transmission.
        """
        n = len(times)
        gains = np.ones((n, len(self.bands)))
        muffle = np.zeros(n)
        dynamic_zones = [z for z in self.zones if z.dynamic]
        if not dynamic_zones:
            return gains, muffle
        source_idx = np.asarray(source_idx)
        times_a = np.asarray(times)
        source_pos = np.asarray(source_pos)
        for si in set(source_idx.tolist()):
            if si < 0:
                continue
            m = source_idx == si
            k = int(m.sum())
            pts = [np.atleast_2d(source_pos[m]),
                   np.broadcast_to(self.rooms[si].center, (1, 3))]
            for zone in dynamic_zones:
                hit = np.zeros(k, dtype=bool)
                for a, bpt in zip(pts[:-1], pts[1:]):
                    hit |= zone.segments_cross(a, bpt)
                if not hit.any():
                    continue
                zg, zm = zone.material_at(times_a[m])
                gains[m] = np.where(hit[:, None], gains[m] * zg, gains[m])
                muffle[m] = np.where(
                    hit, 1.0 - (1.0 - muffle[m]) * (1.0 - zm), muffle[m]
                )
        return gains, muffle

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
        nb = len(self.bands)
        gain = np.ones((n, nb))
        mix = np.zeros(n)
        diff = listener_idx != source_idx
        if not diff.any():
            if self.zones and listener_pos is not None and source_pos is not None:
                zg, _ = self._zone_factors(
                    [source_pos, listener_pos], times=times,
                )
                gain = gain * zg
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
            k = int(m.sum())
            g_sum = np.zeros(k)
            bg_sum = np.zeros((k, nb))
            g_weighted_mix = np.zeros(k)
            for path in paths:
                open_prod = np.ones(k)
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
                    zg, _ = self._zone_factors(
                        waypoints, times=times[m]
                    )
                else:
                    zg = np.ones((k, nb))
                g_sum += g_path
                bg_sum += g_path[:, None] * zg
                g_weighted_mix += g_path * mix_path
            total = np.minimum(g_sum, 1.0)[:, None]
            bg = bg_sum / np.maximum(g_sum, 1e-12)[:, None]
            floor = self.DIRECT_NO_PATH_GAIN
            gain[m] = np.maximum(np.minimum(total, 1.0) * bg, floor)
            mix[m] = np.where(
                g_sum > 1e-12, g_weighted_mix / np.maximum(g_sum, 1e-12), 1.0
            )
        return gain, mix

    def reverb_transmission(self, listener_idx, room_i, times, listener_pos=None):
        """Per-sample band gains at which room_i's reverb is heard.

        Multi-hop rooms contribute along their strongest portal path, so a
        far room's tail is still audible (fainter) through open doors.
        Zones crossed by the strongest listener -> room-centre polyline
        damp each band of the audible tail independently.
        """
        n = len(times)
        nb = len(self.bands)
        gain = np.full((n, nb), self.REVERB_NO_PATH_GAIN)
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
            k = int(m.sum())
            best = np.zeros(k)
            best_zg = np.ones((k, nb))
            for path in paths:
                open_prod = np.ones(k)
                for portal in path:
                    open_prod = open_prod * portal.motion.open_fraction(times[m])
                g_path = self.REVERB_CLOSED_GAIN ** len(path) + (
                    (1.0 - self.REVERB_CLOSED_GAIN) ** len(path)
                ) * open_prod
                zg = np.ones((k, nb))
                if self.zones and listener_pos is not None:
                    waypoints = [listener_pos[m]] + [
                        self._portal_position(portal) for portal in path
                    ] + [self.rooms[room_i].center]
                    zg, _ = self._zone_factors(
                        waypoints, times=times[m], dynamic_filter=False
                    )
                take = g_path > best
                best = np.where(take, g_path, best)
                best_zg = np.where(take[:, None], zg, best_zg)
            floor = self.REVERB_NO_PATH_GAIN
            gain[m] = np.maximum(best[:, None] * best_zg, floor)
        return gain

    def path_report(self, listener_pos, source_pos, t=0.0):
        """Describe every propagation path at time t for path tracing.

        Returns a list of dicts, strongest path first::

            {"kind": "direct"|"detour", "doors": [name, ...],
             "open": [fraction, ...], "door_gain": float,
             "zones": [ {"name", "bands": {band: gain}}, ... ],
             "band_gains": {band: total gain}}

        For same-room points the single path has no doors. Materials are
        those actually installed at time t.
        """
        listener_pos = np.atleast_2d(np.asarray(listener_pos, dtype=np.float64))
        source_pos = np.atleast_2d(np.asarray(source_pos, dtype=np.float64))
        li = int(self.room_index_at(listener_pos)[0])
        si = int(self.room_index_at(source_pos)[0])
        times = np.array([float(t)])
        reports = []
        if li == si:
            paths = [[]]
        elif li < 0 or si < 0:
            paths = []
        else:
            paths = self._simple_paths(li, si)
        for path in paths:
            opens = [float(p.motion.open_fraction(t)) for p in path]
            g_door = 1.0
            for ofr in opens:
                g_door *= self.DIRECT_CLOSED_GAIN + (
                    1.0 - self.DIRECT_CLOSED_GAIN
                ) * ofr
            zone_info = []
            band_gain = {b: 1.0 for b in self.bands}
            if self.zones:
                if path:
                    waypoints = [listener_pos] + [
                        np.atleast_2d(self._portal_position(p)) for p in path
                    ] + [source_pos]
                else:
                    waypoints = [source_pos, listener_pos]
                pts = [np.broadcast_to(w, (1, 3)) for w in waypoints]
                for zone in self.zones:
                    hit = False
                    for a, bpt in zip(pts[:-1], pts[1:]):
                        hit = hit or bool(zone.segments_cross(a, bpt)[0])
                    if not hit:
                        continue
                    zg, _ = zone.material_at(times)
                    bands = {
                        b: float(zg[0, j]) for j, b in enumerate(self.bands)
                    }
                    zone_info.append({"name": zone.name, "bands": bands})
                    for b in self.bands:
                        band_gain[b] *= bands[b]
            reports.append({
                "kind": "direct" if len(path) <= 1 else "detour",
                "doors": [p.name for p in path],
                "open": opens,
                "door_gain": float(g_door),
                "zones": zone_info,
                "band_gains": band_gain,
            })
        reports.sort(key=lambda r: -r["door_gain"])
        return reports

    def to_dict(self):
        return {
            "rooms": [r.to_dict() for r in self.rooms],
            "portals": [p.to_dict() for p in self.portals],
            "zones": [z.to_dict() for z in self.zones],
            "bands": list(self.bands),
            "band_edges": list(self.band_edges),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            [Room.from_dict(r) for r in d.get("rooms", [])],
            [Portal.from_dict(p) for p in d.get("portals", [])],
            [AbsorptionZone.from_dict(z) for z in d.get("zones", [])],
            bands=tuple(d.get("bands", DEFAULT_BANDS)),
            band_edges=tuple(d.get("band_edges", DEFAULT_BAND_EDGES)),
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
