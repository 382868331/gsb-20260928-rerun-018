from __future__ import annotations

from .model import Vec3, Orientation


class Listener:
    """Listener state that can be updated independently of sources."""

    def __init__(
        self,
        position: Vec3 | tuple[float, float, float] | None = None,
        orientation: Orientation | None = None,
    ):
        self.position = position if isinstance(position, Vec3) else Vec3(*(position or (0, 0, 0)))
        self.orientation = orientation or Orientation()

    def update(
        self,
        position: Vec3 | tuple[float, float, float] | None = None,
        orientation: Orientation | None = None,
    ) -> None:
        if position is not None:
            self.position = position if isinstance(position, Vec3) else Vec3(*position)
        if orientation is not None:
            self.orientation = orientation
