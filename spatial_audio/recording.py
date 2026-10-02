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
from .engine import SpatialEngine, RenderResult
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
    def from_dict(cls, data):
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

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


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


def render_interval(recording: MotionRecording, t0: float, t1: float,
                    block_size: int = 1024) -> RenderResult:
    """Render only [t0, t1); identical to truncating a full render.

    State is advanced sample-exactly from t=0 and only the requested
    window is kept; the same soft limiter as a full render is applied to
    the window (the limiter is memoryless, so this matches truncation).
    """
    from .engine import RenderStats

    engine = build_engine(recording)
    sr = recording.sample_rate
    n0 = int(round(t0 * sr))
    n1 = int(round(t1 * sr))
    out = np.zeros((max(0, n1 - n0), 2), dtype=np.float64)
    pos, t = 0.0, 0.0
    sample_pos = 0
    while sample_pos < n1:
        n = min(int(block_size), n1 - sample_pos)
        block, _ = engine.render_block(t, n)
        a, b = max(sample_pos, n0), min(sample_pos + n, n1)
        if a < b:
            out[a - n0:b - n0] = block[a - sample_pos:b - sample_pos]
        sample_pos += n
        t += n / sr
    pre_peak = float(np.max(np.abs(out))) if out.size else 0.0
    limited, max_red = engine._limit(out)
    post_peak = float(np.max(np.abs(limited))) if limited.size else 0.0
    stats = RenderStats(
        duration=float(t1 - t0),
        sample_rate=sr,
        pre_limiter_peak=pre_peak,
        post_limiter_peak=post_peak,
        max_gain_reduction_db=(
            float(20.0 * np.log10(pre_peak / max(post_peak, 1e-12)))
            if max_red > 1e-12 else 0.0
        ),
        limited=max_red > 1e-12,
    )
    return RenderResult(samples=limited, sample_rate=sr, stats=stats)


def save_session(path, recording: MotionRecording, engine, time_seconds: float):
    """Save a recording plus the engine's carried state at time_seconds.

    Reloading restores the room scheme (including absorption zones and
    door motions), every trajectory, and the reverb tails that were still
    decaying, so listening can continue exactly where it stopped.
    """
    data = {
        "recording": recording.to_dict(),
        "time": float(time_seconds),
        "state": engine.snapshot(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_session(path):
    """Load a saved session -> (recording, engine, time_seconds).

    The returned engine already carries the saved state; continue with
    engine.render(remaining, start_time=time_seconds).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    recording = MotionRecording.from_dict(data["recording"])
    engine = build_engine(recording)
    engine.restore_state(data["state"])
    return recording, engine, float(data["time"])
