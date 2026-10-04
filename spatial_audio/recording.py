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
from .rooms import Panel, RoomScene


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


def _clone_scene(scene: RoomScene) -> RoomScene:
    return RoomScene.from_dict(scene.to_dict())


def _find_panel(scene: RoomScene, panel_name: str) -> Panel:
    matches = [p for p in scene.panels if p.name == panel_name]
    if not matches:
        raise KeyError('scene has no panel ' + repr(panel_name))
    return matches[0]


def render_panel_candidate(recording: MotionRecording, panel_name: str,
                           candidate: str, start=None, end=None,
                           block_size: int = 1024, *,
                           boundary=None, mode: str = "history",
                           reset_reverbs=None):
    """Render an A/B preview for one overlapping panel candidate."""
    if recording.scene is None:
        raise ValueError("recording has no room scene")
    scene = _clone_scene(recording.scene)
    panel = _find_panel(scene, panel_name)
    revision = panel.revisions[panel._revision_index(candidate)]
    start = revision.start if start is None else float(start)
    end = revision.end if end is None else float(end)
    start = max(start, revision.start)
    end = min(end, revision.end)
    if not start < end:
        raise ValueError("preview interval does not overlap the candidate")
    boundary = start if boundary is None else float(boundary)
    if boundary < 0.0 or boundary > start:
        raise ValueError("boundary must be in [0, candidate interval start]")
    if mode not in {"history", "recomputed"}:
        raise ValueError("mode must be history or recomputed")
    panel._set_decision(start, end, candidate)
    engine = build_engine(recording, scene=scene)
    if reset_reverbs is None:
        reset_reverbs = mode == "recomputed"
    if boundary > 0.0:
        engine._render_span(0.0, boundary, block_size)
    if reset_reverbs:
        engine.reset_reverb_buses()
    result = engine._render_span(boundary, end - boundary, block_size)
    result.output_mode = "recomputed" if reset_reverbs else "history"
    result.source_boundary = boundary
    result.source_interval = {"start": start, "end": end}
    result.source_version = candidate
    result.tail_source = "discarded" if reset_reverbs else "preserved"
    return result


def adopt_panel_revision(recording: MotionRecording, panel_name: str,
                         candidate: str, start=None, end=None):
    """Persist a user explicit partial candidate choice."""
    if recording.scene is None:
        raise ValueError("recording has no room scene")
    panel = _find_panel(recording.scene, panel_name)
    return panel.select_revision(candidate, start, end)


def withdraw_panel_revision(recording: MotionRecording, panel_name: str,
                            candidate: str = None, start=None, end=None):
    """Withdraw a choice without deleting the source candidate."""
    if recording.scene is None:
        raise ValueError("recording has no room scene")
    panel = _find_panel(recording.scene, panel_name)
    panel.unselect_revision(candidate, start, end)
    return {
        "panel": panel_name,
        "remaining_conflicts": panel.remaining_conflicts(recording.duration),
    }


def remaining_panel_conflicts(recording: MotionRecording, scene=None):
    """Return unresolved overlapping candidate spans for every panel."""
    used_scene = scene if scene is not None else recording.scene
    if used_scene is None:
        return []
    out = []
    for panel in used_scene.panels:
        conflicts = panel.remaining_conflicts(recording.duration)
        if conflicts:
            out.append({"panel": panel.name, "conflicts": conflicts})
    return out


def panel_version_manifest(recording: MotionRecording, start=0.0, end=None,
                           scene=None):
    """Run-length panel-version trace for every sample in a render span.

    Each segment names the panel version actually used at those sample
    times ("base" or a PanelRevision name), its placement/heading and the
    effective per-band transmission. This makes an audible
    render auditable without inferring geometry from the output audio.
    """
    used_scene = scene if scene is not None else recording.scene
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
        versions = panel.version_at(times)
        candidates = panel.candidates_at(times)
        decisions = panel.decision_names_at(times)
        positions, headings = panel.pose_at(times)
        gains = panel.band_gains_at(times)
        segments = []
        k = 0
        while k < n:
            j = k + 1
            while (j < n and versions[j] == versions[k]
                   and list(candidates[j]) == list(candidates[k])
                   and decisions[j] == decisions[k]):
                j += 1
            segment = {
                "version": str(versions[k]),
                "candidate_versions": [str(v) for v in candidates[k]],
                "decision": None if decisions[k] is None else str(decisions[k]),
                "unresolved_conflict": len(candidates[k]) > 1 and decisions[k] is None,
                "sample_start": k,
                "sample_end": j,
                "time_start": float(times[k]),
                "time_end": float(times[j - 1]) + 1.0 / sr,
                "position": [float(v) for v in positions[k]],
                "heading": float(headings[k]),
            }
            if gains is not None:
                segment["band_gains"] = [float(v) for v in gains[k]]
            segments.append(segment)
            k = j
        manifest["panels"].append({
            "name": panel.name,
            "segments": segments,
            "remaining_conflicts": panel.remaining_conflicts(end),
            "decisions": [
                {"start": float(a), "end": float(b), "revision": str(name)}
                for a, b, name in panel.decisions
            ],
        })
    return manifest


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
