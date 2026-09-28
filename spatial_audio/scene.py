"""Scene objects: listener state and sound emitters, independently updateable."""

from __future__ import annotations

import numpy as np

from .resources import SoundResource
from .trajectory import Trajectory, Waypoint


def _as_vec3(value):
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,):
        raise ValueError("expected a length-3 vector")
    return v


class Listener:
    """A listener at a position facing forward, with an up vector.

    State can be updated at any time; it never belongs to an emitter.
    """

    def __init__(self, position=(0.0, 0.0, 0.0), forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0),
                 trajectory=None):
        self.position = _as_vec3(position)
        self.set_orientation(forward, up)
        self.trajectory = trajectory

    def set_position(self, position):
        self.position = _as_vec3(position)

    def set_orientation(self, forward, up=(0.0, 0.0, 1.0)):
        f = _as_vec3(forward)
        u = _as_vec3(up)
        if np.linalg.norm(f) < 1e-12:
            raise ValueError("forward direction must be non-zero")
        self.forward = f / np.linalg.norm(f)
        right = np.cross(self.forward, u)
        if np.linalg.norm(right) < 1e-12:
            raise ValueError("up must not be parallel to forward")
        self.right = right / np.linalg.norm(right)

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory


class Emitter:
    """A sounding object.

    Motion is a continuous Trajectory; sound is a SoundResource. Both can be
    replaced independently at any time. Deactivation is smoothed by the
    engine (no clicks).
    """

    def __init__(
        self,
        name,
        resource,
        trajectory=None,
        position=(0.0, 0.0, 0.0),
        volume=1.0,
        active=True,
        loop=False,
    ):
        self.name = str(name)
        self._resource = resource
        if trajectory is None:
            trajectory = Trajectory([Waypoint(0.0, position)])
        self.trajectory = trajectory
        self.volume = float(volume)
        self.active = bool(active)
        self.loop = bool(loop)
        self.start_time = 0.0

    @property
    def resource(self):
        return self._resource

    def set_resource(self, resource, restart=False, engine_time=None):
        """Swap the sound independently of motion/listener state."""
        self._resource = resource
        if restart:
            if engine_time is None:
                raise ValueError("engine_time required to restart playback")
            self.start_time = float(engine_time)

    def set_active(self, active):
        self.active = bool(active)
