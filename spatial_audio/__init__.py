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
    load_checkpoint,
    panel_version_manifest,
    render_recording,
    save_checkpoint,
)
from .rooms import (
    AbsorptionZone,
    DoorMotion,
    Panel,
    PanelRevision,
    Portal,
    Room,
    RoomScene,
)

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
    "panel_version_manifest",
    "save_checkpoint",
    "load_checkpoint",
    "DoorMotion",
    "Portal",
    "Room",
    "RoomScene",
    "AbsorptionZone",
    "Panel",
    "PanelRevision",
]
