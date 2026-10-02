"""Corridor/multi-room paths, absorption zones, snapshots and intervals."""

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
    load_session,
    render_interval,
    render_recording,
    save_session,
)
from spatial_audio.recording import EmitterSpec, build_engine

SR = 48000


def _tone(freq=500.0, duration=3.0):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.005)


def _corridor_rooms(reverb_send=0.25):
    return [
        Room("corridor", (-12.0, -1.5, -2.0), (0.0, 1.5, 2.0),
             reverb_time=0.5, reverb_send=reverb_send),
        Room("room1", (0.0, -4.0, -2.0), (6.0, 4.0, 2.0),
             reverb_time=0.8, reverb_send=reverb_send),
        Room("room2", (6.0, -4.0, -2.0), (12.0, 4.0, 2.0),
             reverb_time=1.0, reverb_send=reverb_send),
        Room("room3", (0.0, -9.0, -2.0), (12.0, -5.0, 2.0),
             reverb_time=0.6, reverb_send=reverb_send),
    ]


def _corridor_scene(door_b_keys=((0.0, 1.0),), door_c_keys=((0.0, 1.0),),
                    zones=(), detour=True, reverb_send=0.25):
    portals = [
        Portal("door_a", "corridor", "room1", DoorMotion([(0.0, 1.0)])),
        Portal("door_b", "room1", "room2", DoorMotion(list(door_b_keys))),
    ]
    if detour:
        portals += [
            Portal("door_c", "room1", "room3", DoorMotion(list(door_c_keys))),
            Portal("door_d", "room3", "room2", DoorMotion([(0.0, 1.0)])),
        ]
    return RoomScene(_corridor_rooms(reverb_send), portals, list(zones))


def _static_engine(scene, source_pos, listener_pos=(10.0, 2.0, 0.0),
                   freq=500.0, duration=3.0):
    engine = SpatialEngine(
        Listener(position=listener_pos, forward=(-1.0, 0.0, 0.0)),
        sample_rate=SR, scene=scene,
    )
    engine.add_emitter(Emitter("src", _tone(freq, duration), position=source_pos))
    return engine


def _level(samples, t0, t1):
    seg = samples[int(t0 * SR):int(t1 * SR)]
    return float(np.sqrt(np.mean(seg ** 2)))


def test_multi_hop_chain_each_door_matters():
    # corridor -> room1 -> room2, no detour: closing either door on the
    # only route must strongly attenuate the far-room source.
    src = (-6.0, 0.0, 0.0)
    both_open = _static_engine(_corridor_scene(detour=False), src)
    ref = both_open.render(1.5, block_size=256).samples
    b_closed = _static_engine(
        _corridor_scene(door_b_keys=[(0.0, 0.0)], detour=False), src
    ).render(1.5, block_size=256).samples
    scene_a_closed = _corridor_scene(detour=False)
    scene_a_closed.portals[0].motion = DoorMotion([(0.0, 0.0)])
    a_closed = _static_engine(scene_a_closed, src).render(1.5, block_size=256).samples
    open_level = _level(ref, 0.5, 1.4)
    assert _level(b_closed, 0.5, 1.4) < 0.3 * open_level
    assert _level(a_closed, 0.5, 1.4) < 0.3 * open_level


def test_detour_path_survives_closed_direct_door():
    # Loop topology: door_b (direct) closed, detour via room3 open.
    src = (-6.0, 0.0, 0.0)
    detour = _static_engine(
        _corridor_scene(door_b_keys=[(0.0, 0.0)]), src
    ).render(1.5, block_size=256).samples
    # All routes blocked: door_b and door_c closed.
    blocked = _static_engine(
        _corridor_scene(door_b_keys=[(0.0, 0.0)], door_c_keys=[(0.0, 0.0)]),
        src,
    ).render(1.5, block_size=256).samples
    detour_level = _level(detour, 0.5, 1.4)
    blocked_level = _level(blocked, 0.5, 1.4)
    assert detour_level > 3.0 * blocked_level
    assert blocked_level > 1e-6  # leakage floor, not digital silence


def _zone():
    return AbsorptionZone("treatment", (1.5, -1.0, -2.0), (4.5, 1.0, 2.0),
                          absorption=0.75)


def test_absorption_zone_only_affects_crossing_paths():
    # No reverb: isolate the direct path.
    kwargs = dict(detour=False, reverb_send=0.0)
    plain = _corridor_scene(**kwargs)
    treated = _corridor_scene(zones=[_zone()], **kwargs)

    # Path corridor -> room2 crosses the zone: quieter by ~ (1 - absorption).
    src_cross = (-6.0, 0.0, 0.0)
    a = _static_engine(plain, src_cross).render(1.2, block_size=256).samples
    b = _static_engine(treated, src_cross).render(1.2, block_size=256).samples
    ratio = _level(b, 0.4, 1.1) / _level(a, 0.4, 1.1)
    assert 0.2 < ratio < 0.4

    # Path inside room2 misses the zone: bit-identical output.
    src_near = (9.0, 0.0, 0.0)
    c = _static_engine(_corridor_scene(**kwargs), src_near)
    d = _static_engine(_corridor_scene(zones=[_zone()], **kwargs), src_near)
    np.testing.assert_array_equal(
        c.render(1.2, block_size=256).samples,
        d.render(1.2, block_size=256).samples,
    )


def _hf_energy(samples, t0, t1, cutoff=2000.0):
    seg = samples[int(t0 * SR):int(t1 * SR), 0]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
    return float(spec[freqs > cutoff].sum())


def test_absorption_distinguishable_from_door_closing():
    # Absorption is pure gain (spectrum preserved); a closing door also
    # lowpasses (spectrum tilts). The two must remain separable.
    kwargs = dict(detour=False, reverb_send=0.0)
    src = (-6.0, 0.0, 0.0)
    base = _static_engine(_corridor_scene(**kwargs), src, freq=3000.0)
    base_s = base.render(1.2, block_size=256).samples
    treated = _static_engine(
        _corridor_scene(zones=[_zone()], **kwargs), src, freq=3000.0
    ).render(1.2, block_size=256).samples
    door_shut = _static_engine(
        _corridor_scene(door_b_keys=[(0.0, 0.0)], **kwargs), src, freq=3000.0
    ).render(1.2, block_size=256).samples

    hf_base = _hf_energy(base_s, 0.4, 1.1)
    hf_treated = _hf_energy(treated, 0.4, 1.1)
    hf_door = _hf_energy(door_shut, 0.4, 1.1)
    # Absorption scales the 3 kHz tone by the zone factor only.
    assert 0.2 < hf_treated / hf_base < 0.4
    # A closed door removes far more high-frequency energy (muffling).
    assert hf_door / hf_base < 0.1
    # Overall level change of the absorption is smaller than the door's.
    assert _level(treated, 0.4, 1.1) > 2.0 * _level(door_shut, 0.4, 1.1)


def _walking_recording(zones=(), duration=3.0):
    scene = _corridor_scene(
        door_b_keys=[(0.0, 1.0), (1.0, 1.0), (1.5, 0.0), (2.0, 0.0), (2.5, 1.0)],
        zones=zones,
    )
    return MotionRecording(
        duration=duration,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear(
            [(0.0, (10.0, 2.0, 0.0)), (duration, (9.0, 1.5, 0.0))]
        ),
        listener_forward=(-1.0, 0.0, 0.0),
        scene=scene,
        emitters=[
            EmitterSpec(
                name="walker",
                kind="tone",
                params={"frequency": 500.0, "duration": duration, "fade": 0.005},
                trajectory=Trajectory.linear(
                    [(0.0, (-8.0, 0.0, 0.0)), (1.5, (3.0, 0.0, 0.0)),
                     (duration, (9.0, 0.0, 0.0))]
                ),
                events=[(2.2, False)],
            )
        ],
    )


def test_snapshot_resume_bit_identical_and_tail_kept():
    rec = _walking_recording(zones=[_zone()])
    full = render_recording(rec, block_size=1024).samples

    engine = build_engine(rec)
    first = engine.render(1.3, block_size=512)
    state = engine.snapshot()
    engine2 = build_engine(rec)
    engine2.restore_state(state)
    second = engine2.render(rec.duration - 1.3, block_size=384, start_time=1.3)
    resumed = np.concatenate([first.samples, second.samples], axis=0)
    np.testing.assert_allclose(resumed, full, atol=1e-12)
    # The source stopped at 2.2 s; the resumed part still carries the
    # decaying reverb tail.
    tail = resumed[int(2.4 * SR):int(2.7 * SR)]
    assert np.sqrt(np.mean(tail ** 2)) > 1e-5


def test_render_interval_matches_full_render_slice():
    rec = _walking_recording(zones=[_zone()])
    full = render_recording(rec, block_size=1024).samples
    window = render_interval(rec, 0.7, 2.3, block_size=333).samples
    ref = full[int(round(0.7 * SR)):int(round(2.3 * SR))]
    np.testing.assert_allclose(window, ref, atol=1e-9)


def test_session_roundtrip_preserves_scene_and_tail(tmp_path):
    rec = _walking_recording(zones=[_zone()])
    full = render_recording(rec, block_size=1024).samples

    engine = build_engine(rec)
    engine.render(1.1, block_size=256)
    path = tmp_path / "session.json"
    save_session(path, rec, engine, 1.1)

    rec2, engine2, t0 = load_session(path)
    assert abs(t0 - 1.1) < 1e-12
    # Room scheme survived: zones, door keyframes, trajectories.
    assert len(rec2.scene.zones) == 1
    assert abs(rec2.scene.zones[0].absorption - 0.75) < 1e-12
    door_b = rec2.scene.portals[1].motion
    assert abs(door_b.open_fraction(1.75) - 0.0) < 1e-9
    assert abs(door_b.open_fraction(0.5) - 1.0) < 1e-9
    pos, _ = rec2.emitters[0].trajectory.sample(0.75)
    pos_ref, _ = rec.emitters[0].trajectory.sample(0.75)
    np.testing.assert_allclose(pos, pos_ref, atol=1e-12)

    rest = engine2.render(rec2.duration - t0, block_size=300, start_time=t0)
    resumed = np.concatenate([full[: int(1.1 * SR)], rest.samples], axis=0)
    np.testing.assert_allclose(resumed, full, atol=1e-12)
    # Still-decaying tail survived the save/load.
    tail = rest.samples[int(1.3 * SR):int(1.6 * SR)]
    assert np.sqrt(np.mean(tail ** 2)) > 1e-5


def test_corridor_block_size_independent():
    rec = _walking_recording(zones=[_zone()], duration=1.6)
    ref = render_recording(rec, block_size=48000).samples
    for bs in (64, 333, 1024, 5000):
        out = render_recording(rec, block_size=bs).samples
        np.testing.assert_allclose(out, ref, atol=1e-10)
