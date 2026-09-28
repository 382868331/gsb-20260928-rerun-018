"""Continuous-time trajectories (centripetal Catmull-Rom splines).

Position and velocity are defined at *any* time, which is what keeps the
render independent of the processing block size: the engine samples the
trajectory per output sample rather than stepping it per block.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Waypoint:
    time: float
    position: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float64))
        if self.position.shape != (3,):
            raise ValueError("waypoint position must be a length-3 vector")


def _catmull_rom(p0, p1, p2, p3, u):
    """Uniform Catmull-Rom: returns positions and tangents at u in [0, 1]."""
    u = np.asarray(u, dtype=np.float64)
    u2 = u * u
    u3 = u2 * u
    pos = (
        0.5
        * (
            2.0 * p1
            + (-p0 + p2) * u[..., None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2[..., None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3[..., None]
        )
    )
    tang = (
        0.5
        * (
            (-p0 + p2)
            + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u[..., None]
            + 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u2[..., None]
        )
    )
    return pos, tang


class Trajectory:
    """Spline through timed 3-D waypoints, held still beyond the ends."""

    def __init__(self, waypoints):
        if not waypoints:
            raise ValueError("at least one waypoint required")
        pts = sorted(waypoints, key=lambda w: w.time)
        if len(pts) >= 2 and any(pts[i + 1].time <= pts[i].time for i in range(len(pts) - 1)):
            raise ValueError("waypoint times must be strictly increasing")
        self._times = np.array([w.time for w in pts], dtype=np.float64)
        self._points = np.stack([w.position for w in pts]).astype(np.float64)

    @property
    def times(self):
        return self._times

    @property
    def points(self):
        return self._points

    @classmethod
    def linear(cls, coords):
        return cls([Waypoint(t, p) for t, p in coords])

    def sample(self, t):
        """Return (positions, velocities) at scalar or array-like time t.

        Velocity is dposition/dt in units per second (segment duration is
        accounted for), usable directly for the Doppler term.
        """
        scalar = np.isscalar(t)
        tv = np.atleast_1d(np.asarray(t, dtype=np.float64))
        pts, times = self._points, self._times

        if len(times) == 1:
            pos = np.broadcast_to(pts[0], (len(tv), 3)).copy()
            vel = np.zeros_like(pos)
            return (pos[0], vel[0]) if scalar else (pos, vel)

        idx = np.searchsorted(times, tv, side="right") - 1
        idx = np.clip(idx, 0, len(times) - 2)
        t0, t1 = times[idx], times[idx + 1]
        dt = t1 - t0
        u = np.clip((tv - t0) / dt, 0.0, 1.0)

        def point(i):
            return pts[np.clip(i, 0, len(pts) - 1)]

        p0, p1, p2, p3 = point(idx - 1), point(idx), point(idx + 1), point(idx + 2)
        pos, tang = _catmull_rom(p0, p1, p2, p3, u)
        vel = tang / dt[:, None]

        before = tv <= times[0]
        after = tv >= times[-1]
        pos[before] = pts[0]
        vel[before] = 0.0
        pos[after] = pts[-1]
        vel[after] = 0.0
        return (pos[0], vel[0]) if scalar else (pos, vel)

    def to_list(self):
        """JSON-friendly (time, x, y, z) rows."""
        return [[float(t), *self._points[i].tolist()] for i, t in enumerate(self._times)]

    @classmethod
    def from_list(cls, rows):
        return cls([Waypoint(r[0], r[1:4]) for r in rows])
