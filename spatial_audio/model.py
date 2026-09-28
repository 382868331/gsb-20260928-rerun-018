from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import numpy as np


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class Orientation:
    """Listener orientation given by a forward vector and an up vector."""

    forward: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    up: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))

    def basis(self):
        f = self.forward.as_array()
        u = self.up.as_array()
        f /= np.linalg.norm(f)
        r = np.cross(f, u)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return f, r, u
