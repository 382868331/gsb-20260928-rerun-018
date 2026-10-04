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
