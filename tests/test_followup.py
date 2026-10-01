"""Follow-up features: multi-room detour paths, absorption zones,
door-arrangement replay, mid-render checkpoints and interval renders.
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


def _tone(freq=440.0, duration=4.0):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.005)


def _rms(samples, t0, t1):
    seg = samples[int(t0 * SR):int(t1 * SR)]
    return float(np.sqrt(np.mean(seg ** 2)))


def _chain_scene(open_ab=1.0, open_bc=1.0):
    """Three rooms in a row: a - b - c, one door per neighbour pair."""
    rooms = [
        Room("a", (-5.0, -5.0, -2.0), (0.0, 5.0, 2.0), reverb_time=0.8),
        Room("b", (0.0, -5.0, -2.0), (5.0, 5.0, 2.0), reverb_time=0.8),
        Room("c", (5.0, -5.0, -2.0), (10.0, 5.0, 2.0), reverb_time=0.8),
    ]
    portals = [
        Portal("ab", "a", "b", DoorMotion([(0.0, open_ab)])),
        Portal("bc", "b", "c", DoorMotion([(0.0, open_bc)])),
    ]
    return RoomScene(rooms, portals)


def _render(scene, source_pos, duration=1.5, listener_pos=(-2.0, 0.0, 0.0)):
    listener = Listener(position=listener_pos, forward=(1.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
    engine.add_emitter(Emitter("src", _tone(440.0, duration), position=source_pos))
    return engine.render(duration, block_size=512).samples


def test_two_hop_path_follows_both_doors():
    # Source in c, listener in a: the only route crosses both doors.
    both_open = _render(_chain_scene(1.0, 1.0), (7.0, 0.0, 0.0))
    one_closed = _render(_chain_scene(0.0, 1.0), (7.0, 0.0, 0.0))
    both_closed = _render(_chain_scene(0.0, 0.0), (7.0, 0.0, 0.0))
    lo = _rms(both_open, 0.5, 1.4)
    mid = _rms(one_closed, 0.5, 1.4)
    hi = _rms(both_closed, 0.5, 1.4)
    assert lo > 5.0 * mid          # both doors open: clearly audible
    assert mid >= 0.5 * hi         # one closed door ~ no-path level


def test_detour_path_keeps_sound_when_direct_door_closes():
    # Triangle a-b-c: source in b, listener in a. Door a-b closes at t=1;
    # the detour a-c-b stays open, so the source remains clearly audible,
    # unlike a scene where a-b is the only route.
    def scene(with_detour):
        rooms = [
            Room("a", (-5.0, -5.0, -2.0), (0.0, 5.0, 2.0), reverb_time=0.4),
            Room("b", (0.0, -5.0, -2.0), (5.0, 5.0, 2.0), reverb_time=0.4),
            Room("c", (-5.0, 5.0, -2.0), (5.0, 10.0, 2.0), reverb_time=0.4),
        ]
        portals = [
            Portal("ab", "a", "b", DoorMotion([(0.0, 1.0), (1.0, 0.0)])),
            Portal("bc", "b", "c", DoorMotion([(0.0, 1.0)])),
        ]
        if with_detour:
            portals.append(Portal("ac", "a", "c", DoorMotion([(0.0, 1.0)])))
        return RoomScene(rooms, portals)

    detour = _render(scene(True), (2.0, 0.0, 0.0), duration=2.0)
    no_detour = _render(scene(False), (2.0, 0.0, 0.0), duration=2.0)
    open_level = _rms(detour, 0.4, 0.9)
    closed_detour = _rms(detour, 1.2, 1.9)
    closed_alone = _rms(no_detour, 1.2, 1.9)
    assert closed_detour > 0.4 * open_level   # detour keeps it loud
    assert closed_detour > 3.0 * closed_alone  # clearly not a vanished path


def _zone_scene(with_zone):
    rooms = [
        Room("a", (-5.0, -5.0, -2.0), (0.0, 5.0, 2.0), reverb_time=0.5),
        Room("b", (0.0, -5.0, -2.0), (5.0, 5.0, 2.0), reverb_time=0.5),
    ]
    portals = [Portal("door", "a", "b", DoorMotion([(0.0, 1.0)]),
                      position=(0.0, 0.0, 0.0))]
    zones = []
    if with_zone:
        # Absorbing patch in a corner of room b, around one source spot.
        zones.append(AbsorptionZone("panel", (3.0, 3.0, -2.0), (4.0, 4.0, 2.0),
                                    gain=0.3, muffle=0.4))
    return RoomScene(rooms, portals, zones)


def test_absorption_zone_only_damps_crossing_paths():
    plain = _zone_scene(False)
    treated = _zone_scene(True)
    # Path to the corner source crosses the panel: quieter.
    cross_plain = _render(plain, (3.5, 3.5, 0.0))
    cross_treated = _render(treated, (3.5, 3.5, 0.0))
    ratio = _rms(cross_treated, 0.5, 1.4) / _rms(cross_plain, 0.5, 1.4)
    assert 0.2 < ratio < 0.8
    # Path to a source away from the panel is bit-identical: the treatment
    # does not make the whole scene uniformly quieter.
    free_plain = _render(plain, (1.0, -3.0, 0.0))
    free_treated = _render(treated, (1.0, -3.0, 0.0))
    np.testing.assert_allclose(free_plain, free_treated, atol=1e-12)


def test_absorption_distinct_from_door_closing():
    # Source stops at t=1.5. With absorption, the in-room reverb tail is
    # barely affected (the reverb path avoids the panel); closing the door
    # instead strongly suppresses the tail heard next door. The two kinds
    # of change must be clearly distinguishable.
    def render_tail(scene):
        listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
        engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
        e = Emitter("src", _tone(440.0, 3.0), position=(3.5, 3.5, 0.0))
        engine.add_emitter(e)
        engine.schedule_active(e, 1.5, False)
        return engine.render(3.0, block_size=512).samples

    plain = _zone_scene(False)
    tail_plain = _rms(render_tail(plain), 2.0, 2.6)
    tail_treated = _rms(render_tail(_zone_scene(True)), 2.0, 2.6)

    closed_scene = _zone_scene(False)
    closed_scene.portals[0].motion = DoorMotion([(0.0, 1.0), (1.0, 0.0)])
    tail_closed = _rms(render_tail(closed_scene), 2.0, 2.6)

    assert tail_treated > 0.5 * tail_plain   # tail survives treatment
    assert tail_closed < 0.7 * tail_treated  # door closing is much stronger


def test_checkpoint_resume_matches_uninterrupted_render(tmp_path):
    scene = _chain_scene(1.0, 0.0)
    rec = MotionRecording(
        duration=4.0,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear([(0.0, (-2.0, 0.0, 0.0))]),
        listener_forward=(1.0, 0.0, 0.0),
        scene=scene,
        emitters=[
            EmitterSpec(
                name="walker",
                kind="tone",
                params={"frequency": 520.0, "duration": 4.0, "fade": 0.005},
                trajectory=Trajectory.linear(
                    [(0.0, (7.0, 0.0, 0.0)), (4.0, (1.0, 1.0, 0.0))]
                ),
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

    # Chunks concatenate to the same signal as one full render.
    full = render_recording(rec, block_size=1024).samples
    np.testing.assert_allclose(
        np.concatenate([first, second]), full, atol=1e-12
    )
    # The reverb tail was still decaying at the checkpoint and is preserved.
    assert _rms(resumed, 0.2, 0.7) > 1e-6
    # Room scheme and trajectories survive the round trip.
    assert [r.name for r in rec2.scene.rooms] == ["a", "b", "c"]
    assert abs(rec2.emitters[0].trajectory.sample(4.0)[0][0] - 1.0) < 1e-9


def test_render_interval_matches_full_render_slice():
    scene = _chain_scene(1.0, 1.0)

    def build():
        listener = Listener(
            forward=(1.0, 0.0, 0.0),
            trajectory=Trajectory.linear(
                [(0.0, (-2.0, 0.0, 0.0)), (3.0, (-1.0, 1.0, 0.0))]
            ),
        )
        engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
        engine.add_emitter(Emitter("src", _tone(440.0, 3.0), position=(7.0, 0.0, 0.0)))
        return engine

    full = build().render(3.0, block_size=1024).samples
    interval = build().render_interval(1.0, 2.5, block_size=512).samples
    np.testing.assert_allclose(
        interval, full[int(1.0 * SR):int(2.5 * SR)], atol=1e-12
    )


def test_replay_with_different_door_arrangement(tmp_path):
    scene_a = _chain_scene(1.0, 1.0)
    rec = MotionRecording(
        duration=2.0,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear([(0.0, (-2.0, 0.0, 0.0))]),
        listener_forward=(1.0, 0.0, 0.0),
        scene=scene_a,
        emitters=[
            EmitterSpec(
                name="src",
                kind="tone",
                params={"frequency": 440.0, "duration": 2.0, "fade": 0.005},
                trajectory=Trajectory.linear([(0.0, (7.0, 0.0, 0.0))]),
            )
        ],
    )
    path = tmp_path / "take.json"
    rec.save(path)
    loaded = MotionRecording.load(path)

    # Same motion take, different door schedule supplied at replay time.
    scene_b = _chain_scene(1.0, 1.0)
    scene_b.portals[1].motion = DoorMotion([(0.0, 1.0), (1.0, 0.0)])
    out_a = render_recording(loaded, block_size=333).samples
    out_b = render_recording(loaded, block_size=333, scene=scene_b).samples
    assert _rms(out_a, 1.2, 1.9) > 3.0 * _rms(out_b, 1.2, 1.9)
    # The override matches rendering a recording built with that scene.
    np.testing.assert_allclose(
        out_b, render_recording(loaded.with_scene(scene_b), block_size=1024).samples,
        atol=1e-12,
    )


def test_zones_survive_recording_round_trip(tmp_path):
    scene = _zone_scene(True)
    rec = MotionRecording(
        duration=1.0,
        sample_rate=SR,
        scene=scene,
        emitters=[
            EmitterSpec(
                name="src",
                kind="tone",
                params={"frequency": 440.0, "duration": 1.0, "fade": 0.005},
                trajectory=Trajectory.linear([(0.0, (3.5, 3.5, 0.0))]),
            )
        ],
    )
    path = tmp_path / "zoned.json"
    rec.save(path)
    loaded = MotionRecording.load(path)
    assert len(loaded.scene.zones) == 1
    zone = loaded.scene.zones[0]
    assert zone.name == "panel" and abs(zone.gain - 0.3) < 1e-12
    np.testing.assert_allclose(
        render_recording(rec, block_size=500).samples,
        render_recording(loaded, block_size=1024).samples,
        atol=1e-12,
    )
