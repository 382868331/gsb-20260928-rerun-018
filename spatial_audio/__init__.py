"""Spatial sound engine (basic version).

Independent pieces:

- :class:`~spatial_audio.resources.SoundResource`: immutable sample data,
  shared by any number of emitters and independently updateable.
- :class:`~spatial_audio.scene.Emitter`: a sounding object (state updateable).
- :class:`~spatial_audio.scene.Listener`: listener position/orientation
  (state updateable independently of the emitters).
- :class:`~spatial_audio.engine.SpatialEngine`: renders the scene by
  advancing continuous time; chunk size never affects the result.
"""

from .resources import SoundResource
from .trajectory import Trajectory, Waypoint
from .scene import Emitter, Listener
from .engine import SpatialEngine, RenderResult
from .recording import (
    MotionRecording,
    load_session,
    render_interval,
    render_recording,
    save_session,
)
from .rooms import AbsorptionZone, DoorMotion, Portal, Room, RoomScene

__all__ = [
    "SoundResource",
    "Trajectory",
    "Waypoint",
    "Emitter",
    "Listener",
    "SpatialEngine",
    "RenderResult",
    "MotionRecording",
    "render_recording",
    "render_interval",
    "save_session",
    "load_session",
    "AbsorptionZone",
    "DoorMotion",
    "Portal",
    "Room",
    "RoomScene",
]
