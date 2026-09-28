from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


class Trajectory:
    def sample(self, t: float):
        """Return (position as np.ndarray, velocity as np.ndarray) at time t."""
        raise NotImplementedError

    def to_keyframes(self) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class LinearPath(Trajectory):
    """Piecewise-linear path through (t, x, y, z) keyframes. Position is
    continuous in time and velocity is constant on each segment."""

    keyframes: list[tuple[float, float, float, float]]

    def __post_init__(self):
        pts = sorted(self.keyframes)
        if len(pts) < 1:
            raise ValueError("LinearPath needs at least one keyframe")
        self.t = np.array([p[0] for p in pts], dtype=np.float64)
        self.p = np.array([p[1:] for p in pts], dtype=np.float64)

    def sample(self, t: float):
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        if i < 0:
            return self.p[0].copy(), np.zeros(3)
        if i >= len(self.t) - 1:
            return self.p[-1].copy(), np.zeros(3)
        dt = self.t[i + 1] - self.t[i]
        v = (self.p[i + 1] - self.p[i]) / dt
        alpha = (t - self.t[i]) / dt
        return self.p[i] + alpha * (self.p[i + 1] - self[i]), v

    def to_keyframes(self):
        return [[float(self.t[i]), *map(float, self.p[i])] for i in range(len(self.t))]


class TrajectoryRecorder:
    """Persist a motion description (keyframes) to JSON for later replay."""

    @staticmethod
    def save_json(path, frames: list[list[float]] | LinearPath) -> None:
        kf = frames.to_keyframes() if isinstance(frames, LinearPath) else frames
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"keyframes": kf}, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path) -> LinearPath:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return LinearPath([tuple(row) for row in data["keyframes"]])
