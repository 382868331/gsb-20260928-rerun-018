from __future__ import annotations

import numpy as np

from .model import Vec3
from .signals import SignalSource
from .trajectory import Trajectory


class SoundSource:
    """A sounding object. Its sound resource, motion and on/off state can be
    updated independently."""

    def __init__(
        self,
        signal: SignalSource | None = None,
        position: Vec3 | tuple[float, float, float] = (0, 0, 0),
        trajectory: Trajectory | None = None,
        gain: float = 1.0,
        active: bool = True,
        name: str = "source",
        deactivate_at: float | None = None,
    ):
        self.signal = signal
        self.position = position if isinstance(position, Vec3) else Vec3(*position)
        self.trajectory = trajectory
        self.gain = float(gain)
        self.active = bool(active)
        self.name = name
        # If set, the engine deactivates the source at this continuous time.
        # Storing it in the source keeps abrupt-stop events replayable.
        self.deactivate_at = deactivate_at

    def bind_signal(self, signal: SignalSource | None) -> None:
        self.signal = signal

    def update(self, position=None, trajectory: Trajectory | None = ..., gain=None,
               active: bool | None = None) -> None:
        if position is not None:
            self.position = position if isinstance(position, Vec3) else Vec3(*position)
        if trajectory is not ...:
            self.trajectory = trajectory
        if gain is not None:
            self.gain = float(gain)
        if active is not None:
            self.active = bool(active)

    def pose_at(self, t: float):
        if self.trajectory is None:
            return self.position.as_array(), np.zeros(3)
        return self.trajectory.sample(t)
