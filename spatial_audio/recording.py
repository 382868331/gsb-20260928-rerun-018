"""Save a motion scene and replay (re-render) it deterministically.

The recording stores trajectories, tones and activation events as JSON;
replaying reconstructs the exact scene so the exported audio is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np

from .resources import SoundResource
from .trajectory import Trajectory
from .scene import Emitter, Listener
from .engine import SpatialEngine, RenderResult, RenderStats
from .rooms import RoomScene


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
                 listener_forward=(0.0, 1.0, 0.0), emitters=None, scene=None):
        self.duration = float(duration)
        self.sample_rate = int(sample_rate)
        self.listener_trajectory = listener_trajectory
        self.listener_forward = tuple(listener_forward)
        self.emitters = list(emitters or [])
        self.scene = scene

    def to_dict(self):
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "listener_forward": list(self.listener_forward),
            "listener_trajectory":
                self.listener_trajectory.to_list() if self.listener_trajectory else None,
            "scene": self.scene.to_dict() if self.scene is not None else None,
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
        scene_data = data.get("scene")
        scene = RoomScene.from_dict(scene_data) if scene_data else None
        return cls(data["duration"], data["sample_rate"], lt,
                   data.get("listener_forward", (0.0, 1.0, 0.0)), specs,
                   scene=scene)


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
    engine = SpatialEngine(listener, sample_rate=recording.sample_rate,
                           scene=recording.scene)
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


def render_recording_resumed(recording: MotionRecording, state, block_size=1024,
                             duration=None):
    """Build an engine, restore a suspended state and render on.

    ``state`` is a dict from ``SpatialEngine.suspend_state`` (or the JSON
    loaded from a state file). Rendering continues for ``duration``
    seconds (default: the recording's remaining duration), so a take saved
    mid-way and continued has the same total length and samples as one
    continuously rendered take.
    """
    engine = build_engine(recording)
    engine.restore_state(state)
    remaining = recording.duration - engine.current_time
    if duration is None:
        duration = remaining
    return engine.render(duration, block_size=block_size)


def recording_with_doors(recording: MotionRecording, door_motions):
    """Copy a recording with new door states, keeping all motion/events.

    ``door_motions`` maps portal name to :class:`DoorMotion`. Source
    trajectories, resources, volume and start/stop events are untouched,
    so different door openings can be auditioned against one recorded
    take. The returned recording is a separate object.
    """
    if recording.scene is None:
        raise ValueError("recording has no room scene / doors")
    new_scene = recording.scene.with_door_motions(door_motions)
    return MotionRecording(
        duration=recording.duration,
        sample_rate=recording.sample_rate,
        listener_trajectory=recording.listener_trajectory,
        listener_forward=recording.listener_forward,
        emitters=list(recording.emitters),
        scene=new_scene,
    )


def concat_results(first: RenderResult, second: RenderResult) -> RenderResult:
    """Concatenate two consecutive raw renders of the same take.

    Inputs should be pre-limiter (e.g. halves of one take rendered with
    suspended state); concatenation therefore matches one continuous raw
    render before the safety limiter.
    """
    if first.sample_rate != second.sample_rate:
        raise ValueError("sample rates must match")
    samples = np.concatenate([first.samples, second.samples], axis=0)
    sr = first.sample_rate
    pre_peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    stats = RenderStats(
        duration=len(samples) / sr,
        sample_rate=sr,
        pre_limiter_peak=pre_peak,
        post_limiter_peak=pre_peak,
        max_gain_reduction_db=0.0,
        limited=False,
        emitter_peaks={**first.stats.emitter_peaks, **second.stats.emitter_peaks},
    )
    return RenderResult(samples=samples, sample_rate=sr, stats=stats)
