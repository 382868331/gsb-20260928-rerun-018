"""Render an artificial tone moving along a trajectory, export it, and verify
the spatial result and the engine's continuous-time guarantees."""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from spatial_audio import (
    LinearPath,
    Listener,
    Scenario,
    SineTone,
    SoundSource,
    SpatialEngine,
    Vec3,
)
from spatial_audio.export import export_wav

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SR = 48000
DURATION = 7.0


def build_scenario() -> Scenario:
    listener = Listener(position=Vec3(0, 0, 0))
    moving = SoundSource(
        signal=SineTone(frequency=440.0, amplitude=0.6),
        trajectory=LinearPath([
            (0.0, -6.0, 4.0, 0.0),
            (6.0, 6.0, 4.0, 0.0),
        ]),
        name="moving_tone",
    )
    beeper = SoundSource(
        signal=SineTone(frequency=660.0, amplitude=0.5),
        position=Vec3(-2.0, 0.5, 0.0),
        name="beeper",
        deactivate_at=2.5,
    )
    return Scenario(DURATION, SR, listener, [moving, beeper])


def render(scenario: Scenario, chunk: int | None = None) -> tuple[np.ndarray, SpatialEngine]:
    engine = SpatialEngine(
        sample_rate=scenario.sample_rate,
        listener=scenario.listener,
        sources=scenario.sources,
    )
    n = int(scenario.duration * scenario.sample_rate)
    out = []
    produced = 0
    block = chunk or n
    while produced < n:
        cur = min(block, n - produced)
        out.append(engine.process(cur))
        produced += cur
    return np.concatenate(out, axis=0), engine


def window_rms(audio: np.ndarray, seconds: float = 0.25) -> list[dict]:
    w = int(seconds * SR)
    rows = []
    for start in range(0, len(audio) - w + 1, w):
        seg = audio[start:start + w]
        rms_l = float(np.sqrt(np.mean(seg[:, 0] ** 2)))
        rms_r = float(np.sqrt(np.mean(seg[:, 1] ** 2)))
        rows.append({
            "t": round(start / SR, 3),
            "rms_left": round(rms_l, 5),
            "rms_right": round(rms_r, 5),
            "dominant": "left" if rms_l > rms_r * 1.05 else
                        ("right" if rms_r > rms_l * 1.05 else "center"),
        })
    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    audio, engine = render(build_scenario())

    rows = window_rms(audio)
    with open(os.path.join(OUT_DIR, "channel_levels.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    meta = export_wav(os.path.join(OUT_DIR, "moving_source.wav"), audio, SR)
    scenario_path = os.path.join(OUT_DIR, "scenario.json")
    digest = build_scenario().save(scenario_path)

    a1, _ = render(build_scenario(), chunk=37)
    a2, _ = render(build_scenario(), chunk=256)
    a3, _ = render(build_scenario(), chunk=4096)
    chunk_diff = float(max(np.max(np.abs(a1 - a2)), np.max(np.abs(a1 - a3))))

    replay_audio, replay_engine = render(Scenario.load(scenario_path), chunk=512)
    replay_diff = float(np.max(np.abs(replay_audio - a2)))

    stop = int(2.5 * SR)
    tail = audio[stop:stop + int(0.05 * SR)]
    max_jump = float(np.max(np.abs(np.diff(tail, axis=0))))
    post_energy = float(np.sqrt(np.mean(audio[stop + int(0.05 * SR):stop + int(0.1 * SR)] ** 2)))

    early = rows[1]
    late = rows[int(5.5 / 0.25)]
    summary = {
        "wav": meta,
        "scenario_sha256": digest,
        "channel_check": {
            "early_dominant": early["dominant"],
            "late_dominant": late["dominant"],
            "early_rms": [early["rms_left"], early["rms_right"]],
            "late_rms": [late["rms_left"], late["rms_right"]],
        },
        "chunk_size_invariance_max_abs_diff": chunk_diff,
        "saved_scenario_replay_max_abs_diff": replay_diff,
        "deactivation_max_sample_step": max_jump,
        "post_tail_rms": post_energy,
        "report": {
            "peak_pre_limiter": engine.report.peak_pre_limiter,
            "over_level_events": engine.report.over_level_events,
            "stopped_sources": engine.report.stopped_sources,
            "notes": engine.report.notes[:4],
        },
        "replay_stopped_sources": replay_engine.report.stopped_sources,
    }
    with open(os.path.join(OUT_DIR, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
