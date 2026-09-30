"""Save a motion scene and replay (re-render) it deterministically.

The recording stores trajectories, tones and activation events as JSON;
replaying reconstructs the exact scene so the exported audio is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .resources import SoundResource
from .trajectory import Trajectory
from .scene import Emitter, Listener
from .engine import SpatialEngine, RenderResult
from .rooms import Door, OCCLUDER_CLASSES, Occluder, Room


@dataclass
class EmitterSpec:
    name: str
    kind: str  # "tone"
    params: dict
    trajectory: Trajectory
    volume: float = 1.0
    loop: bool = False
    events: list | None = None  # [(time, active)]


class MotionRecording:
    """Serializable description of a whole spatial motion take."""

    def __init__(self, duration, sample_rate, listener_trajectory=None,
                 listener_forward=(0.0, 1.0, 0.0), emitters=None,
                 rooms=None, occluders=None, listener_position=None):
        self.duration = float(duration)
        self.sample_rate = int(sample_rate)
        self.listener_trajectory = listener_trajectory
        self.listener_forward = tuple(listener_forward)
        self.emitters = list(emitters or [])
        self.rooms = list(rooms or [])
        self.occluders = list(occluders or [])
        self.listener_position = listener_position

    def to_dict(self):
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "listener_forward": list(self.listener_forward),
            "listener_position": (
                list(self.listener_position)
                if self.listener_position is not None else None
            ),
            "listener_trajectory":
                self.listener_trajectory.to_list() if self.listener_trajectory else None,
            "rooms": [r.to_dict() for r in self.rooms],
            "occluders": [w.to_dict() for w in self.occluders],
            "emitters": [
                {
                    "name": e.name,
                    "kind": e.kind,
                    "params": e.params,
                    "trajectory": e.trajectory.to_list(),
                    "volume": e.volume,
                    "loop": e.loop,
                    "events": e.events or [],
                }
                for e in self.emitters
            ],
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lt = Trajectory.from_list(data["listener_trajectory"]) if data.get("listener_trajectory") else None
        specs = [
            EmitterSpec(
                name=e["name"],
                kind=e["kind"],
                params=e["params"],
                trajectory=Trajectory.from_list(e["trajectory"]),
                volume=e.get("volume", 1.0),
                loop=e.get("loop", False),
                events=[tuple(ev) for ev in e.get("events", [])],
            )
            for e in data["emitters"]
        ]
        rooms = [Room.from_dict(d) for d in data.get("rooms", [])]
        occluders = [OCCLUDER_CLASSES[d.get("kind", "wall")].from_dict(d)
                     for d in data.get("occluders", [])]
        return cls(data["duration"], data["sample_rate"], lt,
                   data.get("listener_forward", (0.0, 1.0, 0.0)), specs,
                   rooms=rooms, occluders=occluders,
                   listener_position=data.get("listener_position"))


def _build_resource(kind, params, sample_rate):
    if kind == "tone":
        return SoundResource.sine_tone(
            frequency=params["frequency"],
            duration=params["duration"],
            sample_rate=sample_rate,
            phase=params.get("phase", 0.0),
            fade=params.get("fade", 0.01),
        )
    raise ValueError(f"unsupported resource kind: {kind}")


def build_engine(recording: MotionRecording) -> SpatialEngine:
    listener = Listener(forward=recording.listener_forward,
                        trajectory=recording.listener_trajectory)
    if recording.listener_trajectory is not None:
        listener.position = recording.listener_trajectory.points[0]
    elif recording.listener_position is not None:
        listener.set_position(recording.listener_position)
    engine = SpatialEngine(
        listener,
        sample_rate=recording.sample_rate,
        rooms=recording.rooms,
        occluders=recording.occluders,
    )
    for spec in recording.emitters:
        resource = _build_resource(spec.kind, spec.params, recording.sample_rate)
        initially_active = True
        events = sorted(spec.events or [])
        if events and events[0][0] <= 0.0:
            initially_active = bool(events[0][1])
        emitter = Emitter(
            name=spec.name,
            resource=resource,
            trajectory=spec.trajectory,
            volume=spec.volume,
            loop=spec.loop,
            active=initially_active,
        )
        engine.add_emitter(emitter)
        for t, active in events:
            engine.schedule_active(emitter, float(t), bool(active))
    return engine


def render_recording(recording: MotionRecording, block_size: int = 1024) -> RenderResult:
    engine = build_engine(recording)
    return engine.render(recording.duration, block_size=block_size)
