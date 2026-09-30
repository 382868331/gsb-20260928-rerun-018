"""Rooms, occluding walls and door portals for propagation geometry.

Acoustics are deliberately low-dimensional (the motion lives in a 2-D floor
plan, height z is ignored for geometry):

- A :class:`Room` is an axis-aligned box; a position belongs to exactly one
  room and membership is blended near the shared wall so a listener walking
  through a doorway hears a continuous crossfade.
- An :class:`Occluder` is a wall segment. A direct path crossing the segment
  is diffracted around its edges: the excess path length maps smoothly to
  extra attenuation and a low-pass cutoff, with *zero* change the moment the
  ray only grazes the wall edge -- so a source passing through an open door
  changes volume and spectrum continuously.
- A :class:`Door` is a wall segment with an openness schedule. When shut it
  blocks the direct path (strong attenuation / filtering); when ajar its
  blocking fraction interpolates with the openness. Scheduled openness is
  linearly ramped so closing a door never clicks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _as_xy(value):
    v = np.asarray(value, dtype=np.float64)
    if v.shape == (3,):
        v = v[:2]
    if v.shape != (2,):
        raise ValueError("expected a length-2 (or length-3) point")
    return v


@dataclass
class Room:
    """Axis-aligned rectangular room with shared reflective acoustics."""

    name: str
    min_xy: tuple
    max_xy: tuple
    rt60: float = 0.45
    reverb_level: float = 0.18

    def __post_init__(self):
        lo = _as_xy(self.min_xy)
        hi = _as_xy(self.max_xy)
        if np.any(hi <= lo):
            raise ValueError("room max_xy must exceed min_xy on both axes")
        self.min_xy = lo
        self.max_xy = hi
        self.rt60 = float(self.rt60)
        self.reverb_level = float(self.reverb_level)

    def contains(self, xy):
        xy = _as_xy(xy)
        return bool(np.all(xy >= self.min_xy) and np.all(xy <= self.max_xy))

    def to_dict(self):
        return {
            "name": self.name,
            "min_xy": [float(x) for x in self.min_xy],
            "max_xy": [float(x) for x in self.max_xy],
            "rt60": self.rt60,
            "reverb_level": self.reverb_level,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            min_xy=d["min_xy"],
            max_xy=d["max_xy"],
            rt60=d.get("rt60", 0.45),
            reverb_level=d.get("reverb_level", 0.18),
        )


@dataclass
class Occluder:
    """A solid wall segment on the 2-D floor plan."""

    name: str
    a: tuple
    b: tuple
    # Closed-wall attenuation applied to a path pushed far around the edge.
    max_attenuation_db: float = 24.0
    min_cutoff_hz: float = 400.0
    # Excess path length (m) that reaches max attenuation.
    diffraction_scale: float = 2.0

    def __post_init__(self):
        self.a = _as_xy(self.a)
        self.b = _as_xy(self.b)
        if np.linalg.norm(self.b - self.a) < 1e-9:
            raise ValueError("occluder endpoints must differ")

    def is_door(self):
        return False

    def to_dict(self):
        return {
            "kind": "wall",
            "name": self.name,
            "a": [float(x) for x in self.a],
            "b": [float(x) for x in self.b],
            "max_attenuation_db": self.max_attenuation_db,
            "min_cutoff_hz": self.min_cutoff_hz,
            "diffraction_scale": self.diffraction_scale,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            a=d["a"],
            b=d["b"],
            max_attenuation_db=d.get("max_attenuation_db", 24.0),
            min_cutoff_hz=d.get("min_cutoff_hz", 400.0),
            diffraction_scale=d.get("diffraction_scale", 2.0),
        )


@dataclass
class Door(Occluder):
    """A wall segment whose openness can change over time.

    openness 1.0 = fully open (no blocking), 0.0 = shut. Events are
    (time, openness) pairs; values between events are linearly ramped so a
    moving door never causes a gain/spectrum jump.
    """

    openness: float = 1.0
    closed_attenuation_db: float = 20.0
    closed_cutoff_hz: float = 500.0
    events: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.openness = float(self.openness)
        self.closed_attenuation_db = float(self.closed_attenuation_db)
        self.closed_cutoff_hz = float(self.closed_cutoff_hz)
        self.events = [(float(t), float(o)) for t, o in self.events]

    def is_door(self):
        return True

    def set_openness(self, value):
        self.openness = float(np.clip(value, 0.0, 1.0))

    def schedule_openness(self, time_seconds, openness):
        self.events.append((float(time_seconds), float(np.clip(openness, 0.0, 1.0))))

    def openness_at(self, times):
        """Piecewise-linear openness, held outside the scheduled range."""
        t = np.asarray(times, dtype=np.float64)
        out = np.full_like(t, self.openness)
        events = sorted(self.events)
        if not events:
            return out
        et = np.array([e[0] for e in events])
        ev = np.array([e[1] for e in events])
        idx = np.searchsorted(et, t)
        before = idx == 0
        after = idx >= len(events)
        mid = ~(before | after)
        out[after] = ev[-1]
        if np.any(mid):
            k = np.clip(idx[mid] - 1, 0, len(events) - 2)
            t0, t1 = et[k], et[k + 1]
            frac = np.zeros_like(t0)
            gap = t1 - t0
            nz = gap > 1e-12
            frac[nz] = np.clip((t[mid][nz] - t0[nz]) / gap[nz], 0.0, 1.0)
            # Equal-time events step immediately.
            frac[~nz] = 1.0
            out[mid] = ev[k] + (ev[k + 1] - ev[k]) * frac
        return np.clip(out, 0.0, 1.0)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "kind": "door",
            "openness": self.openness,
            "closed_attenuation_db": self.closed_attenuation_db,
            "closed_cutoff_hz": self.closed_cutoff_hz,
            "events": [list(e) for e in self.events],
        })
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            a=d["a"],
            b=d["b"],
            max_attenuation_db=d.get("max_attenuation_db", 24.0),
            min_cutoff_hz=d.get("min_cutoff_hz", 400.0),
            diffraction_scale=d.get("diffraction_scale", 2.0),
            openness=d.get("openness", 1.0),
            closed_attenuation_db=d.get("closed_attenuation_db", 20.0),
            closed_cutoff_hz=d.get("closed_cutoff_hz", 500.0),
            events=[tuple(e) for e in d.get("events", [])],
        )


OCCLUDER_CLASSES = {"wall": Occluder, "door": Door}


def segment_intersections(p0, p1, a, b):
    """Vectorized segment crossing parameters along p0 -> p1.

    Returns (hit, u) where u is the position along the source->listener ray
    at the crossing; hit is False for parallel / non-crossing rays.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    w = a - p0
    seg = b - a
    det = d[:, 0] * seg[1] - d[:, 1] * seg[0]
    hit = np.abs(det) > 1e-12
    safe = np.where(hit, det, 1.0)
    w_x = a[0] - p0[:, 0]
    w_y = a[1] - p0[:, 1]
    u = (w_x * seg[1] - w_y * seg[0]) / safe
    v = (d[:, 1] * w_x - d[:, 0] * w_y) / safe
    crossed = hit & (u > 0.0) & (u < 1.0) & (v >= 0.0) & (v <= 1.0)
    return crossed, u, v


def room_weights(positions, rooms, margin=0.35):
    """Per-room soft membership weights for an array of 2-D positions.

    Weights sum to 1 for every position. Up to two rooms share weight near a
    boundary (within ``margin`` metres); positions outside all rooms snap to
    the nearest room centre.
    """
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    n = len(xy)
    nr = len(rooms)
    if nr == 0:
        return np.zeros((n, 0))
    lo = np.stack([r.min_xy for r in rooms])
    hi = np.stack([r.max_xy for r in rooms])
    center = 0.5 * (lo + hi)
    inside = np.all((xy[:, None, :] >= lo[None]) & (xy[:, None, :] <= hi[None]), axis=2)
    dist_c = np.linalg.norm(xy[:, None, :] - center[None, :, :], axis=2)

    w = np.zeros((n, nr))
    counts = inside.sum(axis=1)
    nearest = np.argmin(dist_c, axis=1)
    rows = np.arange(n)
    w[rows, nearest[rows]] = np.where(counts == 0, 1.0, 0.0)
    multi = counts > 1
    for rj in range(nr):
        m = multi & inside[:, rj]
        w[m, rj] = 1.0 / counts[m]
    single = counts == 1
    only = np.argmax(inside, axis=1)
    w[single, only[single]] = 1.0

    if nr >= 2:
        # Blend toward the room on the other side of each nearby axis-aligned
        # boundary, within the margin -- continuous doorway crossing.
        blend = np.zeros((n, nr))
        for ri, room in enumerate(rooms):
            for rj, other in enumerate(rooms):
                if ri == rj:
                    continue
                for axis in (0, 1):
                    a1 = 1 - axis
                    span = (max(room.min_xy[a1], other.min_xy[a1]),
                            min(room.max_xy[a1], other.max_xy[a1]))
                    if span[1] <= span[0]:
                        continue
                    along = (xy[:, a1] >= span[0]) & (xy[:, a1] <= span[1])
                    d2 = np.full(n, -1.0)
                    if abs(other.max_xy[axis] - room.min_xy[axis]) < 1e-9:
                        # other room is on the low side of room.min
                        d2 = xy[:, axis] - room.min_xy[axis]
                    elif abs(other.min_xy[axis] - room.max_xy[axis]) < 1e-9:
                        d2 = room.max_xy[axis] - xy[:, axis]
                    near = along & (d2 >= 0.0) & (d2 <= margin)
                    if np.any(near):
                        s = 0.5 - 0.5 * np.cos(np.pi * (1.0 - d2[near] / margin))
                        blend[near, rj] = np.maximum(blend[near, rj], 0.5 * s)
        moved_rows = blend.sum(axis=1) > 0
        total = blend.sum(axis=1, keepdims=True)
        w[moved_rows] = w[moved_rows] * (1.0 - total[moved_rows]) + blend[moved_rows]
    return w
