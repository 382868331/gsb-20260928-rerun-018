"""Save a motion scene and replay (re-render) it deterministically.

The recording stores trajectories, tones and activation events as JSON;
replaying reconstructs the exact scene so the exported audio is reproducible.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Optional

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

    def with_scene(self, scene):
        """Copy of this recording with a different room/door arrangement.

        Trajectories, tones and events are shared unchanged, so the same
        motion take can be replayed under another door schedule or with
        added absorption zones.
        """
        return MotionRecording(
            self.duration, self.sample_rate, self.listener_trajectory,
            self.listener_forward, self.emitters, scene=scene,
        )


def _build_resource(kind, params, sample_rate):
    if kind == "tone":
        return SoundResource.sine_tone(
            frequency=params["frequency"],
            duration=params["duration"],
            sample_rate=sample_rate,
            phase=params.get("phase", 0.0),
            fade=params.get("fade", 0.01),
        )
    if kind == "mix_tones":
        n = int(round(params["duration"] * sample_rate))
        tt = np.arange(n) / sample_rate
        signal = np.zeros(n)
        for frequency, amplitude in params["tones"]:
            signal += float(amplitude) * np.sin(2.0 * np.pi * float(frequency) * tt)
        fade_n = int(round(params.get("fade", 0.01) * sample_rate))
        if fade_n > 0:
            ramp = 0.5 - 0.5 * np.cos(
                np.pi * np.clip(np.arange(fade_n) / max(fade_n - 1, 1), 0.0, 1.0)
            )
            signal[:fade_n] *= ramp
            signal[-fade_n:] *= ramp[::-1]
        return SoundResource(signal, sample_rate)
    raise ValueError(f"unsupported resource kind: {kind}")


def build_engine(recording: MotionRecording, scene=None) -> SpatialEngine:
    listener = Listener(forward=recording.listener_forward,
                        trajectory=recording.listener_trajectory)
    if recording.listener_trajectory is not None:
        listener.position = recording.listener_trajectory.points[0]
    engine = SpatialEngine(listener, sample_rate=recording.sample_rate,
                           scene=scene if scene is not None else recording.scene)
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


def render_recording(recording: MotionRecording, block_size: int = 1024,
                     scene=None) -> RenderResult:
    """Render a recording; `scene` overrides the stored door arrangement."""
    engine = build_engine(recording, scene=scene)
    return engine.render(recording.duration, block_size=block_size)


def panel_version_manifest(recording: MotionRecording, start=0.0, end=None,
                           scene=None, preview=None):
    """Run-length panel-version trace for every sample in a render span.

    Each segment names the panel version actually used at those sample
    times ("base" or a PanelRevision name), its placement/heading and the
    effective per-band transmission. This makes an audible
    render auditable without inferring geometry from the output audio.
    """
    used_scene = scene if scene is not None else recording.scene
    if preview is not None:
        used_scene = copy.deepcopy(used_scene)
        panel_name, candidate_name = preview[0], preview[1]
        preview_start = preview[2] if len(preview) > 2 else None
        preview_end = preview[3] if len(preview) > 3 else None
        panel_index = next(
            i for i, panel in enumerate(used_scene.panels)
            if panel.name == panel_name
        )
        used_scene.panels[panel_index] = used_scene.panels[panel_index].preview_revision(
            candidate_name, preview_start, preview_end
        )
    sr = recording.sample_rate
    start = float(start)
    end = float(recording.duration if end is None else end)
    n = max(0, int(round((end - start) * sr)))
    times = start + (np.arange(n) + 0.5) / sr
    manifest = {
        "sample_rate": sr,
        "start": start,
        "end": end,
        "band_edges": list(used_scene.band_edges),
        "panels": [],
    }
    for panel in used_scene.panels:
        versions, statuses, candidate_rows = panel.selection_report_at(times)
        positions, headings = panel.pose_at(times)
        gains = panel.band_gains_at(times)
        segments = []
        k = 0
        while k < n:
            j = k + 1
            while (j < n and versions[j] == versions[k]
                   and statuses[j] == statuses[k]
                   and list(candidate_rows[j] or []) == list(candidate_rows[k] or [])):
                j += 1
            segment = {
                "version": str(versions[k]),
                "sample_start": k,
                "sample_end": j,
                "time_start": float(times[k]),
                "time_end": float(times[j - 1]) + 1.0 / sr,
                "position": [float(v) for v in positions[k]],
                "heading": float(headings[k]),
                "status": str(statuses[k]),
                "candidates": [str(v) for v in (candidate_rows[k] or [])],
                "remaining_conflict": bool(statuses[k] == "conflict"),
            }
            if gains is not None:
                segment["band_gains"] = [float(v) for v in gains[k]]
            segments.append(segment)
            k = j
        manifest["panels"].append({
            "name": panel.name,
            "segments": segments,
            "remaining_conflicts": panel.overlapping_candidates(start, end),
        })
    return manifest


def render_panel_candidate(recording: MotionRecording, panel_name: str,
                           candidate_name: str, block_size: int = 1024,
                           start: Optional[float] = None,
                           end: Optional[float] = None,
                           recompute_from: Optional[float] = None):
    """Render an A/B preview without mutating the saved panel decision.

    The candidate can be tried on a subrange of its interval. By default the
    returned render keeps history; pass recompute_from to produce the explicit
    boundary-recomputed comparison output.
    """
    scene = copy.deepcopy(recording.scene)
    panel_index = next(
        i for i, panel in enumerate(scene.panels)
        if panel.name == panel_name
    )
    panel = scene.panels[panel_index]
    scene.panels[panel_index] = panel.preview_revision(
        candidate_name, start, end
    )
    engine = build_engine(recording, scene=scene)
    if recompute_from is None:
        return engine.render(recording.duration, block_size=block_size)
    boundary = float(recompute_from)
    preview_panel = scene.panels[panel_index]
    override = preview_panel.preview_override
    preview_panel.preview_override = None
    if boundary > 0.0:
        engine._render_span(0.0, boundary, block_size)
    engine.reset_reverb_buses()
    preview_panel.preview_override = override
    duration = recording.duration - boundary
    result = engine._render_span(
        boundary, duration, block_size, output_mode="recomputed",
    )
    engine._clock = boundary + duration
    return result


def save_checkpoint(recording: MotionRecording, engine: SpatialEngine, path):
    """Save recording + full engine state so listening can resume later.

    The snapshot keeps the room scheme, trajectories, event schedule and
    every decaying reverb tail exactly as they are at engine time
    `engine._clock`.
    """
    data = {
        "recording": recording.to_dict(),
        "state": engine.get_state(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_checkpoint(path):
    """Load a checkpoint -> (recording, engine) ready for render_chunk().

    The engine clock and all carried state are restored, so continuing
    with render_chunk() produces exactly the samples that would have
    followed in the original session.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    recording = MotionRecording.from_dict(data["recording"])
    engine = build_engine(recording)
    engine.set_state(data["state"])
    return recording, engine
