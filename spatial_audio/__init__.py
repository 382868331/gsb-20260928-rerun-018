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
from .recording import MotionRecording, render_recording

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
]
