"""Frequency-band absorption: per-band materials, mid-replay material
swaps, multi-zone paths, path tracing, and state consistency.
"""

import numpy as np

from spatial_audio import (
    AbsorptionZone,
    DoorMotion,
    Emitter,
    Listener,
    MotionRecording,
    Portal,
    Room,
    RoomScene,
    SoundResource,
    SpatialEngine,
    Trajectory,
    load_checkpoint,
    render_recording,
    save_checkpoint,
)
from spatial_audio.recording import EmitterSpec, build_engine

SR = 48000
LOW = 300.0    # below the default 1 kHz band edge
HIGH = 4000.0  # above it


def _tone(freq=440.0, duration=4.0):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.005)


def _rms(samples, t0, t1):
    seg = samples[int(t0 * SR):int(t1 * SR)]
    return float(np.sqrt(np.mean(seg ** 2)))


def _band_energy(samples, t0, t1, freq):
    """Energy of the left channel near one frequency in [t0, t1)."""
    seg = samples[int(t0 * SR):int(t1 * SR), 0]
    window = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * window)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
    k = int(np.argmin(np.abs(freqs - freq)))
    return float(spec[max(0, k - 3):k + 4].sum())


def _two_room_scene(zones):
    rooms = [
        Room("a", (-5.0, -5.0, -2.0), (0.0, 5.0, 2.0), reverb_time=0.6),
        Room("b", (0.0, -5.0, -2.0), (5.0, 5.0, 2.0), reverb_time=0.6),
    ]
    portals = [Portal("door", "a", "b", DoorMotion([(0.0, 1.0)]),
                      position=(0.0, 0.0, 0.0))]
    return RoomScene(rooms, portals, zones)


# Panel in a corner of room b; paths to the corner source cross it.
PANEL_CORNER = ((3.0, 3.0, -2.0), (4.0, 4.0, 2.0))
CORNER_SRC = (3.5, 3.5, 0.0)
FREE_SRC = (1.0, -3.0, 0.0)


def _render(scene, source_pos, freq=440.0, duration=1.5, events=None):
    listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
    emitter = Emitter("src", _tone(freq, duration), position=source_pos)
    engine.add_emitter(emitter)
    for t, active in events or []:
        engine.schedule_active(emitter, t, active)
    return engine.render(duration, block_size=512).samples


def test_band_material_damps_high_but_keeps_low():
    # One material: high band almost absorbed, low band nearly transparent.
    panel = AbsorptionZone("panel", *PANEL_CORNER, bands=[0.95, 0.05])
    plain = _two_room_scene([])
    treated = _two_room_scene([panel])

    low_plain = _render(plain, CORNER_SRC, freq=LOW)
    low_treated = _render(treated, CORNER_SRC, freq=LOW)
    high_plain = _render(plain, CORNER_SRC, freq=HIGH)
    high_treated = _render(treated, CORNER_SRC, freq=HIGH)

    low_ratio = _rms(low_treated, 0.5, 1.4) / _rms(low_plain, 0.5, 1.4)
    high_ratio = _rms(high_treated, 0.5, 1.4) / _rms(high_plain, 0.5, 1.4)
    assert low_ratio > 0.7    # lows survive the treatment
    assert high_ratio < 0.3   # highs are strongly absorbed


def test_untreated_path_keeps_original_spectrum():
    panel = AbsorptionZone("panel", *PANEL_CORNER, bands=[0.9, 0.05])
    plain = _two_room_scene([])
    treated = _two_room_scene([panel])
    # A source whose path avoids the panel is bit-identical: the retrofit
    # does not recolor untreated regions.
    free_plain = _render(plain, FREE_SRC, freq=HIGH)
    free_treated = _render(treated, FREE_SRC, freq=HIGH)
    np.testing.assert_allclose(free_plain, free_treated, atol=1e-12)


def test_multifreq_source_band_energies():
    # One sound containing both bands: only the high part is damped.
    n = int(1.5 * SR)
    t = np.arange(n) / SR
    samples = 0.5 * np.sin(2 * np.pi * LOW * t) + 0.5 * np.sin(2 * np.pi * HIGH * t)
    resource = SoundResource(samples, SR)
    panel = AbsorptionZone("panel", *PANEL_CORNER, bands=[0.95, 0.05])

    def render(scene):
        listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
        engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
        engine.add_emitter(Emitter("src", resource, position=CORNER_SRC))
        return engine.render(1.5, block_size=512).samples

    plain = render(_two_room_scene([]))
    treated = render(_two_room_scene([panel]))
    low_ratio = _band_energy(treated, 0.5, 1.4, LOW) / _band_energy(plain, 0.5, 1.4, LOW)
    high_ratio = _band_energy(treated, 0.5, 1.4, HIGH) / _band_energy(plain, 0.5, 1.4, HIGH)
    assert low_ratio > 0.5
    # The one-pole crossover leaks a little 4 kHz into the low band, so
    # the residual is small but not zero.
    assert high_ratio < 0.1


def test_material_swap_mid_replay_is_not_retroactive():
    # High band dull (0.05) until t=2.0, then swapped to clear (0.9).
    swap = AbsorptionZone(
        "panel", *PANEL_CORNER,
        bands=[1.0, [(0.0, 0.05), (2.0, 0.05), (2.2, 0.9)]],
    )
    dull = AbsorptionZone("panel", *PANEL_CORNER, bands=[1.0, 0.05])
    clear = AbsorptionZone("panel", *PANEL_CORNER, bands=[1.0, 0.9])
    # Source plays 0-1.5 s, stops, plays again 2.5-3.5 s.
    events = [(1.5, False), (2.5, True)]
    out_swap = _render(_two_room_scene([swap]), CORNER_SRC, freq=HIGH,
                       duration=3.5, events=events)
    out_dull = _render(_two_room_scene([dull]), CORNER_SRC, freq=HIGH,
                       duration=3.5, events=events)
    out_clear = _render(_two_room_scene([clear]), CORNER_SRC, freq=HIGH,
                        duration=3.5, events=events)

    # The tail decaying after the source stopped was produced while the
    # dull material was in place: swapping materials at t=2 must not
    # recolor it, so it matches the always-dull render exactly.
    np.testing.assert_allclose(
        out_swap[int(1.6 * SR):int(1.95 * SR)],
        out_dull[int(1.6 * SR):int(1.95 * SR)], atol=1e-9)
    assert _rms(out_clear, 1.7, 1.95) > 3.0 * _rms(out_swap, 1.7, 1.95)
    # Sound produced after the swap uses the new (clear) material.
    assert _rms(out_swap, 2.8, 3.4) > 3.0 * _rms(out_dull, 2.8, 3.4)
    assert _rms(out_swap, 2.8, 3.4) > 0.5 * _rms(out_clear, 2.8, 3.4)


def test_crossing_two_panels_differs_from_one():
    one = [AbsorptionZone("p1", *PANEL_CORNER, bands=[0.5, 0.5])]
    two = one + [AbsorptionZone("p2", (3.2, 3.2, -2.0), (3.8, 3.8, 2.0),
                                bands=[0.5, 0.5])]
    out_one = _render(_two_room_scene(one), CORNER_SRC, freq=LOW)
    out_two = _render(_two_room_scene(two), CORNER_SRC, freq=LOW)
    assert _rms(out_two, 0.5, 1.4) < 0.75 * _rms(out_one, 0.5, 1.4)


def test_explain_path_reports_route_and_materials():
    panel = AbsorptionZone("panel", *PANEL_CORNER,
                           bands=[0.9, [(0.0, 0.1), (2.0, 0.8)]])
    scene = _two_room_scene([panel])
    info = scene.explain_path((-2.0, 0.0, 0.0), CORNER_SRC, 1.0)
    assert info["listener_room"] == "a" and info["source_room"] == "b"
    assert [p["name"] for p in info["portals"]] == ["door"]
    assert info["portals"][0]["open"] == 1.0
    assert [z["name"] for z in info["zones"]] == ["panel"]
    assert abs(info["band_gains"][0] - 0.9) < 1e-9
    assert 0.1 < info["band_gains"][1] < 0.8  # mid-interpolation at t=1
    # A source away from the panel crosses no zones.
    free = scene.explain_path((-2.0, 0.0, 0.0), FREE_SRC, 1.0)
    assert free["zones"] == [] and free["band_gains"] is None


def test_broadband_zones_still_work():
    # The original scalar gain/muffle scheme is untouched.
    zone = AbsorptionZone("panel", *PANEL_CORNER, gain=0.3, muffle=0.4)
    assert zone.band_gains_at(0.0) is None
    scene = _two_room_scene([zone])
    assert not scene.has_band_zones
    plain = _render(_two_room_scene([]), CORNER_SRC)
    treated = _render(scene, CORNER_SRC)
    ratio = _rms(treated, 0.5, 1.4) / _rms(plain, 0.5, 1.4)
    assert 0.2 < ratio < 0.8


def test_block_size_independence_with_bands():
    panel = AbsorptionZone("panel", *PANEL_CORNER,
                           bands=[0.9, [(0.0, 0.1), (1.0, 0.7)]])
    scene = _two_room_scene([panel])
    reference = None
    for block in (37, 256, 1024, 10007):
        listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
        engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
        engine.add_emitter(Emitter("src", _tone(HIGH, 1.0), position=CORNER_SRC))
        out = engine.render(1.0, block_size=block).samples
        if reference is None:
            reference = out
        else:
            np.testing.assert_allclose(out, reference, atol=1e-12)


def test_checkpoint_and_interval_with_band_zones(tmp_path):
    panel = AbsorptionZone("panel", *PANEL_CORNER, bands=[0.9, 0.1])
    scene = _two_room_scene([panel])
    rec = MotionRecording(
        duration=4.0,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear([(0.0, (-2.0, 0.0, 0.0))]),
        listener_forward=(1.0, 0.0, 0.0),
        scene=scene,
        emitters=[
            EmitterSpec(
                name="src",
                kind="tone",
                params={"frequency": HIGH, "duration": 4.0, "fade": 0.005},
                trajectory=Trajectory.linear([(0.0, CORNER_SRC)]),
                events=[(1.5, False)],  # tail still decaying at checkpoint
            )
        ],
    )
    engine = build_engine(rec)
    first = engine.render_chunk(2.0).samples
    ckpt = tmp_path / "mid.json"
    save_checkpoint(rec, engine, ckpt)
    second = engine.render_chunk(2.0).samples

    rec2, engine2 = load_checkpoint(ckpt)
    resumed = engine2.render_chunk(2.0).samples
    np.testing.assert_allclose(resumed, second, atol=1e-12)
    full = render_recording(rec, block_size=1024).samples
    np.testing.assert_allclose(np.concatenate([first, second]), full, atol=1e-12)
    assert _rms(resumed, 0.2, 0.7) > 1e-7  # dull tail preserved
    assert rec2.scene.zones[0].bands is not None

    interval = build_engine(rec).render_interval(1.0, 2.5, block_size=512).samples
    np.testing.assert_allclose(interval, full[int(1.0 * SR):int(2.5 * SR)],
                               atol=1e-12)


def test_band_zones_survive_recording_round_trip(tmp_path):
    panel = AbsorptionZone("panel", *PANEL_CORNER,
                           bands=[0.9, [(0.0, 0.1), (2.0, 0.8)]])
    scene = _two_room_scene([panel])
    rec = MotionRecording(
        duration=1.0,
        sample_rate=SR,
        scene=scene,
        emitters=[
            EmitterSpec(
                name="src",
                kind="tone",
                params={"frequency": HIGH, "duration": 1.0, "fade": 0.005},
                trajectory=Trajectory.linear([(0.0, CORNER_SRC)]),
            )
        ],
    )
    path = tmp_path / "bands.json"
    rec.save(path)
    loaded = MotionRecording.load(path)
    zone = loaded.scene.zones[0]
    gains = zone.band_gains_at(np.array([0.0, 2.0]))
    np.testing.assert_allclose(gains[:, 0], [0.9, 0.9], atol=1e-12)
    np.testing.assert_allclose(gains[:, 1], [0.1, 0.8], atol=1e-12)
    np.testing.assert_allclose(
        render_recording(rec, block_size=500).samples,
        render_recording(loaded, block_size=1024).samples, atol=1e-12)
