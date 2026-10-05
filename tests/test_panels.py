"""Moving oriented panels: editable position/heading/band transmission.

The checks use low/mid/high tones and two geometric routes. A panel can
leave one route without erasing its already generated reverb tail, while
a route it never crosses remains bit-identical.
"""

import numpy as np

from spatial_audio import (
    Emitter,
    Listener,
    MotionRecording,
    Panel,
    PanelRevision,
    Room,
    RoomScene,
    SoundResource,
    SpatialEngine,
    Trajectory,
    load_checkpoint,
    panel_version_manifest,
    render_panel_candidate,
    render_recording,
    save_checkpoint,
)
from spatial_audio.recording import EmitterSpec, build_engine

SR = 48000
EDGES = (500.0, 2000.0)
LOW = 200.0
MID = 1000.0
HIGH = 5000.0
ROOM = Room("hall", (-8.0, -4.0, -2.0), (4.0, 4.0, 2.0),
            reverb_time=1.2, reverb_send=0.6)
LISTEN_A = (-6.0, 0.0, 0.0)
SOURCE_A = (0.0, 0.0, 0.0)
LISTEN_B = (-6.0, 3.0, 0.0)
SOURCE_B = (0.0, 3.0, 0.0)
PANEL_POS = (-1.0, 0.0, 0.0)


def _tone(freq, duration):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.003)


def _rms(samples, t0, t1):
    seg = samples[int(t0 * SR):int(t1 * SR)]
    return float(np.sqrt(np.mean(seg ** 2)))


def _panel(moving=False, revised=False):
    position = (
        [(0.0, PANEL_POS), (1.0, PANEL_POS), (2.0, (-1.0, -4.0, 0.0))]
        if moving else [(0.0, PANEL_POS)]
    )
    revisions = []
    if revised:
        revisions.append(PanelRevision(
            "open_high", 1.0, 2.0,
            bands=[None, None, [(1.0, 0.05), (1.4, 0.9)]],
        ))
    return Panel(
        "mover", width=1.2, height=3.0,
        position_keys=position,
        heading_keys=[(0.0, 0.0)],
        bands=[0.9, 0.2, 0.05],
        revisions=revisions,
    )


def _scene(panel=None):
    return RoomScene([ROOM], band_edges=EDGES,
                     panels=[] if panel is None else [panel])


def _conflict_panel():
    panel = _panel()
    panel.propose_revision(PanelRevision(
        "editor_left", 0.8, 2.0,
        position_keys=[(0.0, PANEL_POS)],
        bands=[None, None, 0.75],
    ))
    panel.propose_revision(PanelRevision(
        "editor_right", 0.8, 2.0,
        position_keys=[(0.0, (-1.0, 2.0, 0.0))],
        bands=[None, None, 0.08],
    ))
    return panel


def _conflict_recording(duration=2.2, panel=None):
    return MotionRecording(
        duration=duration, sample_rate=SR,
        listener_trajectory=Trajectory.linear([(0.0, LISTEN_A)]),
        listener_forward=(1.0, 0.0, 0.0),
        scene=_scene(_conflict_panel() if panel is None else panel),
        emitters=[EmitterSpec(
            "src", "tone", {"frequency": HIGH, "duration": duration, "fade": 0.003},
            Trajectory.linear([(0.0, SOURCE_A)]),
        )],
    )


def _render(freq, duration=2.5, panel=None, listener=LISTEN_A,
            source=SOURCE_A, events=None, block_size=512):
    engine = SpatialEngine(
        Listener(position=listener, forward=(1.0, 0.0, 0.0)),
        sample_rate=SR, scene=_scene(panel),
    )
    emitter = Emitter("src", _tone(freq, duration), position=source)
    engine.add_emitter(emitter)
    for event in events or []:
        engine.schedule_active(emitter, *event)
    return engine.render(duration, block_size=block_size).samples


def test_three_band_transmission_and_other_route_untouched():
    static = _panel()
    ratios = {}
    for freq, kept in [(LOW, 0.98), (MID, 0.8), (HIGH, 0.55)]:
        plain = _render(freq, panel=None)
        treated = _render(freq, panel=static)
        ratio = _rms(treated, 0.4, 0.9) / _rms(plain, 0.4, 0.9)
        ratios[freq] = ratio
        assert ratio < kept, f"{freq:g} Hz should be attenuated"
    assert ratios[LOW] > 0.7
    assert ratios[LOW] > ratios[MID] > ratios[HIGH]


    plain_b = _render(HIGH, panel=None, listener=LISTEN_B, source=SOURCE_B)
    treated_b = _render(HIGH, panel=static, listener=LISTEN_B, source=SOURCE_B)
    np.testing.assert_allclose(plain_b, treated_b, atol=1e-12)


def test_moving_panel_stop_does_not_erase_old_tail_and_other_path_survives():
    static_panel = _panel(moving=False)
    moving_panel = _panel(moving=True)
    events = [(0.8, False)]

    static_tail = _render(HIGH, panel=static_panel, events=events)
    moving_tail = _render(HIGH, panel=moving_panel, events=events)
    no_panel_tail = _render(HIGH, panel=None, events=events)

    np.testing.assert_allclose(
        moving_tail[int(1.2 * SR):int(1.8 * SR)],
        static_tail[int(1.2 * SR):int(1.8 * SR)], atol=1e-12,
    )
    assert _rms(moving_tail, 1.2, 1.8) > 1e-5
    assert _rms(no_panel_tail, 1.2, 1.8) > 3.0 * _rms(moving_tail, 1.2, 1.8)
    np.testing.assert_allclose(
        moving_tail[:int(0.9 * SR)], static_tail[:int(0.9 * SR)], atol=1e-12,
    )

    moving_b = _render(HIGH, panel=moving_panel, listener=LISTEN_B,
                       source=SOURCE_B)
    plain_b = _render(HIGH, panel=None, listener=LISTEN_B, source=SOURCE_B)
    np.testing.assert_allclose(moving_b, plain_b, atol=1e-12)


def test_revision_only_changes_selected_interval_and_is_reported():
    dull = _panel()
    revised = _panel(revised=True)
    out_dull = _render(HIGH, duration=2.2, panel=dull)
    out_rev = _render(HIGH, duration=2.2, panel=revised)

    np.testing.assert_allclose(
        out_rev[:int(0.95 * SR)], out_dull[:int(0.95 * SR)], atol=1e-12,
    )
    assert _rms(out_rev, 1.6, 2.1) > 2.0 * _rms(out_dull, 1.6, 2.1)
    mid_dull = _render(MID, duration=2.2, panel=dull)
    mid_rev = _render(MID, duration=2.2, panel=revised)
    before_ratio = _rms(mid_rev, 0.2, 0.9) / _rms(mid_dull, 0.2, 0.9)
    after_ratio = _rms(mid_rev, 1.6, 2.1) / _rms(mid_dull, 1.6, 2.1)
    assert 0.95 < before_ratio < 1.05
    assert 0.7 < after_ratio < 1.05

    rec = MotionRecording(
        duration=2.2, sample_rate=SR,
        listener_trajectory=Trajectory.linear([(0.0, LISTEN_A)]),
        listener_forward=(1.0, 0.0, 0.0), scene=_scene(revised),
        emitters=[EmitterSpec(
            "src", "tone", {"frequency": HIGH, "duration": 2.2, "fade": 0.003},
            Trajectory.linear([(0.0, SOURCE_A)]),
        )],
    )
    manifest = panel_version_manifest(rec)
    segments = manifest["panels"][0]["segments"]
    assert [s["version"] for s in segments] == ["base", "open_high", "base"]
    assert abs(segments[1]["time_start"] - 1.0) < 2.0 / SR
    assert segments[1]["band_gains"][2] > segments[0]["band_gains"][2]


def test_history_vs_recompute_from_boundary_audibly_differ():
    panel = _panel()
    events = [(0.9, False)]
    historical = _render(HIGH, duration=2.2, panel=panel, events=events)

    engine = SpatialEngine(
        Listener(position=LISTEN_A, forward=(1.0, 0.0, 0.0)),
        sample_rate=SR, scene=_scene(panel),
    )
    emitter = Emitter("src", _tone(HIGH, 2.2), position=SOURCE_A)
    engine.add_emitter(emitter)
    engine.schedule_active(emitter, 0.9, False)
    recomputed = engine.render_recomputed_from(1.0, 1.2, reset_reverbs=True).samples

    assert _rms(historical, 1.2, 1.8) > 100.0 * _rms(recomputed, 0.2, 0.8)


def test_block_size_pause_resume_interval_and_recording_agree(tmp_path):
    panel = _panel(moving=True, revised=True)
    rec = MotionRecording(
        duration=1.3, sample_rate=SR,
        listener_trajectory=Trajectory.linear([(0.0, LISTEN_A)]),
        listener_forward=(1.0, 0.0, 0.0), scene=_scene(panel),
        emitters=[EmitterSpec(
            "src", "tone", {"frequency": HIGH, "duration": 1.3, "fade": 0.003},
            Trajectory.linear([(0.0, SOURCE_A)]),
        )],
    )
    full = render_recording(rec, block_size=1024).samples
    for block_size in (1, 37, 256, 10007):
        out = render_recording(rec, block_size=block_size).samples
        np.testing.assert_allclose(out, full, atol=1e-11)

    engine = build_engine(rec)
    first = engine.render_chunk(0.73, block_size=257).samples
    checkpoint = tmp_path / "panel.json"
    save_checkpoint(rec, engine, checkpoint)
    second = engine.render_chunk(0.57, block_size=811).samples
    _, resumed_engine = load_checkpoint(checkpoint)
    resumed = resumed_engine.render_chunk(0.57, block_size=64).samples
    np.testing.assert_allclose(resumed, second, atol=1e-12)
    np.testing.assert_allclose(np.concatenate([first, second]), full, atol=1e-11)

    interval = build_engine(rec).render_interval(0.37, 1.11, block_size=333).samples
    np.testing.assert_allclose(
        interval, full[int(0.37 * SR):int(1.11 * SR)], atol=1e-11,
    )

    path = tmp_path / "rec.json"
    rec.save(path)
    loaded = MotionRecording.load(path)
    roundtrip = render_recording(loaded, block_size=4099).samples
    np.testing.assert_allclose(roundtrip, full, atol=1e-11)
    stored = loaded.scene.panels[0]
    assert stored.name == "mover"
    assert stored.revisions[0].name == "open_high"
    assert len(stored._pos_times) == 3


def test_overlapping_editors_remain_candidates_until_partial_choice():
    rec = _conflict_recording()
    panel = rec.scene.panels[0]

    full = render_recording(rec, block_size=4096).samples
    left_preview = render_panel_candidate(
        rec, "mover", "editor_left", block_size=512,
    ).samples
    right_preview = render_panel_candidate(
        rec, "mover", "editor_right", block_size=512,
    ).samples
    assert _rms(right_preview, 1.1, 1.7) > _rms(left_preview, 1.1, 1.7) * 1.4

    panel.adopt_revision("editor_left", 1.0, 1.5)
    chosen = render_recording(rec, block_size=37).samples
    manifest = panel_version_manifest(rec)
    segments = manifest["panels"][0]["segments"]
    statuses = [(s["version"], s["status"], s["remaining_conflict"]) for s in segments]
    assert ("editor_left", "adopted", False) in statuses
    assert any(item[2] for item in statuses)
    assert manifest["panels"][0]["remaining_conflicts"] == [
        {"start": 0.8, "end": 1.0, "candidates": ["editor_left", "editor_right"]},
        {"start": 1.5, "end": 2.0, "candidates": ["editor_left", "editor_right"]},
    ]

    np.testing.assert_allclose(
        chosen[int(1.0 * SR):int(1.5 * SR)],
        left_preview[int(1.0 * SR):int(1.5 * SR)], atol=1e-11,
    )
    np.testing.assert_allclose(
        chosen[int(1.55 * SR):int(1.9 * SR)],
        full[int(1.55 * SR):int(1.9 * SR)], atol=1e-12,
    )


def test_later_candidate_cannot_replace_previous_partial_decision():
    panel = _conflict_panel()
    panel.adopt_revision("editor_left", 1.0, 1.4)
    panel.propose_revision(PanelRevision(
        "later", 1.05, 1.35, bands=[None, None, 0.01],
    ))
    versions, statuses, candidates = panel.selection_report_at(
        np.array([1.1, 1.55])
    )
    assert versions[0] == "editor_left"
    assert statuses[0] == "adopted"
    assert candidates[0] == ["editor_left", "editor_right", "later"]
    assert versions[1] == "editor_left" and statuses[1] == "conflict"

    panel.reselect_revision("editor_left", "editor_right", 1.1, 1.2)
    versions, statuses, _ = panel.selection_report_at(np.array([1.05, 1.15, 1.3]))
    assert list(versions) == ["editor_left", "editor_right", "editor_left"]
    assert list(statuses) == ["adopted", "adopted", "adopted"]
    panel.withdraw_revision("editor_right")
    versions, statuses, _ = panel.selection_report_at(np.array([1.15]))
    assert versions[0] == "editor_left" and statuses[0] == "conflict"


def test_partial_preview_is_transient_and_round_trip_workflow_stays_consistent(tmp_path):
    rec = _conflict_recording(duration=2.2)
    panel = rec.scene.panels[0]
    panel.adopt_revision("editor_left", 1.0, 1.5)

    formal_before = render_recording(rec, block_size=512).samples
    preview = panel.preview_revision("editor_right", 1.2, 1.4)
    preview_rec = MotionRecording(
        rec.duration, rec.sample_rate, rec.listener_trajectory,
        rec.listener_forward, rec.emitters, scene=_scene(preview),
    )
    preview_audio = render_recording(preview_rec, block_size=37).samples
    api_preview = render_panel_candidate(
        rec, "mover", "editor_right", block_size=37, start=1.2, end=1.4,
    ).samples
    np.testing.assert_allclose(api_preview, preview_audio, atol=1e-12)
    preview_engine = build_engine(preview_rec)
    preview_chunks = np.concatenate([
        preview_engine.render_chunk(1.17, block_size=129).samples,
        preview_engine.render_chunk(0.31, block_size=257).samples,
        preview_engine.render_chunk(0.72, block_size=511).samples,
    ])
    np.testing.assert_allclose(preview_chunks, preview_audio, atol=1e-11)
    np.testing.assert_allclose(
        preview_audio[:int(1.1 * SR)], formal_before[:int(1.1 * SR)], atol=1e-12,
    )
    assert np.max(np.abs(
        preview_audio[int(1.22 * SR):int(1.38 * SR)]
        - formal_before[int(1.22 * SR):int(1.38 * SR)]
    )) > 1e-8

    times = np.array([0.9, 1.1, 1.3, 1.45, 1.7])
    versions, statuses, _ = preview.selection_report_at(times)
    assert list(versions) == [
        "editor_left", "editor_left", "editor_right", "editor_left", "editor_left",
    ]
    assert list(statuses) == [
        "conflict", "adopted", "preview", "adopted", "conflict",
    ]
    assert panel.decisions == [
        {"revision": "editor_left", "start": 1.0, "end": 1.5},
    ]
    assert not hasattr(panel, "preview_decisions") or panel.preview_decisions == []
    assert preview.overlapping_candidates() == panel.overlapping_candidates() == [
        {"start": 0.8, "end": 1.0,
         "candidates": ["editor_left", "editor_right"]},
        {"start": 1.5, "end": 2.0,
         "candidates": ["editor_left", "editor_right"]},
    ]

    panel.reselect_revision("editor_left", "editor_right", 1.2, 1.4)
    panel.withdraw_revision("editor_right", 1.25, 1.35)
    expected_decisions = [
        {"revision": "editor_left", "start": 1.0, "end": 1.2},
        {"revision": "editor_left", "start": 1.4, "end": 1.5},
        {"revision": "editor_right", "start": 1.2, "end": 1.25},
        {"revision": "editor_right", "start": 1.35, "end": 1.4},
    ]
    assert panel.decisions == expected_decisions

    path = tmp_path / "partial-workflow.json"
    rec.save(path)
    reopened = MotionRecording.load(path)
    restored = reopened.scene.panels[0]
    assert restored.decisions == expected_decisions
    assert restored.preview_decisions == []
    assert restored.overlapping_candidates() == [
        {"start": 0.8, "end": 1.0,
         "candidates": ["editor_left", "editor_right"]},
        {"start": 1.25, "end": 1.35,
         "candidates": ["editor_left", "editor_right"]},
        {"start": 1.5, "end": 2.0,
         "candidates": ["editor_left", "editor_right"]},
    ]

    final_full = render_recording(reopened, block_size=1024).samples
    chunk_engine = build_engine(reopened)
    final_chunks = np.concatenate([
        chunk_engine.render_chunk(0.83, block_size=127).samples,
        chunk_engine.render_chunk(1.02, block_size=509).samples,
        chunk_engine.render_chunk(0.35, block_size=2048).samples,
    ])
    np.testing.assert_allclose(final_chunks, final_full, atol=1e-11)

    reopened_preview = restored.preview_revision("editor_right", 1.25, 1.35)
    versions, statuses, _ = reopened_preview.selection_report_at(
        np.array([1.22, 1.3, 1.38])
    )
    assert list(versions) == ["editor_right", "editor_right", "editor_right"]
    assert list(statuses) == ["adopted", "preview", "adopted"]
    assert restored.decisions == expected_decisions


def test_pending_candidates_round_trip_and_history_recompute_are_marked(tmp_path):
    duration = 1.1
    rec = _conflict_recording()
    panel = rec.scene.panels[0]
    rec.duration = duration
    rec.emitters[0].params["duration"] = duration
    panel.adopt_revision("editor_left", 0.82, 0.95)
    path = tmp_path / "pending.json"
    rec.save(path)
    loaded = MotionRecording.load(path)
    restored_panel = loaded.scene.panels[0]
    assert restored_panel.decisions == panel.decisions
    assert [r.name for r in restored_panel.revisions] == [
        "editor_left", "editor_right",
    ]

    full = render_recording(loaded, block_size=257).samples
    for block_size in (37, 64, 4099):
        np.testing.assert_allclose(
            render_recording(loaded, block_size=block_size).samples, full, atol=1e-11,
        )

    historical = render_panel_candidate(loaded, "mover", "editor_right")
    recomputed = render_panel_candidate(
        loaded, "mover", "editor_right", recompute_from=0.8,
    )
    assert historical.mode == "historical"
    assert recomputed.mode == "recomputed"
    assert historical.samples.shape[0] == full.shape[0]
    assert recomputed.samples.shape[0] == int(0.3 * SR)
