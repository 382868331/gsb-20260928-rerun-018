"""Multi-room chains, parallel paths, cross-boundary tails and resume."""

import json

import numpy as np

from spatial_audio import (
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
)
from spatial_audio.recording import (
    EmitterSpec,
    build_engine,
    recording_with_doors,
    render_recording,
    render_recording_resumed,
)

SR = 48000


def _tone(freq=440.0, duration=4.0):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.005)


def _three_rooms(ab_open=1.0, bc_open=1.0, ac_portal=False):
    rooms = [
        Room("a", (-5.0, -5.0, -2.0), (0.0, 5.0, 2.0), reverb_time=0.5),
        Room("b", (0.0, -5.0, -2.0), (5.0, 5.0, 2.0), reverb_time=0.5),
        Room("c", (5.0, -5.0, -2.0), (10.0, 5.0, 2.0), reverb_time=0.5),
    ]
    portals = [
        Portal("ab", "a", "b", DoorMotion([(0.0, ab_open)])),
        Portal("bc", "b", "c", DoorMotion([(0.0, bc_open)])),
    ]
    if ac_portal:
        portals.append(Portal("ac", "a", "c", DoorMotion([(0.0, 0.0)])))
    return RoomScene(rooms, portals)


def _scene_copy(scene):
    return RoomScene(
        scene.rooms,
        [Portal(p.name, p.room_a, p.room_b, p.motion) for p in scene.portals],
    )


def test_sound_propagates_through_room_chain():
    # Source in room c, listener in room a: only path is c -> b -> a.
    listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR, scene=_three_rooms())
    engine.add_emitter(Emitter("src", _tone(440.0, 2.0), position=(7.0, 0.0, 0.0)))
    s = engine.render(1.0, block_size=512).samples
    assert np.sqrt(np.mean(s ** 2)) > 0.05

    # Closing the far door (bc) blocks the whole chain even though the
    # near door (ab) stays open.
    listener2 = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    engine2 = SpatialEngine(
        listener2, sample_rate=SR, scene=_three_rooms(ab_open=1.0, bc_open=0.0)
    )
    engine2.add_emitter(Emitter("src", _tone(440.0, 2.0), position=(7.0, 0.0, 0.0)))
    s2 = engine2.render(1.0, block_size=512).samples
    assert np.sqrt(np.mean(s2 ** 2)) < 0.02


def test_parallel_paths_arrive_together():
    # Two routes a -> c: the direct ac portal and the a -> b -> c detour.
    scene = _three_rooms(ac_portal=True)
    # Direct portal closed: only the muffled 8% leakage would come through,
    # but the chain detour is open and adds sound simultaneously.
    listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
    engine.add_emitter(Emitter("src", _tone(440.0, 2.0), position=(7.0, 0.0, 0.0)))
    s = engine.render(1.0, block_size=256).samples
    level_detour = np.sqrt(np.mean(s ** 2))

    # Same closed direct portal, but chain doors closed too: nearly silent.
    scene_no_detour = _three_rooms(ab_open=0.0, bc_open=0.0, ac_portal=True)
    listener2 = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    engine2 = SpatialEngine(listener2, sample_rate=SR, scene=scene_no_detour)
    engine2.add_emitter(Emitter("src", _tone(440.0, 2.0), position=(7.0, 0.0, 0.0)))
    s2 = engine2.render(1.0, block_size=256).samples
    level_direct_only = np.sqrt(np.mean(s2 ** 2))

    assert level_detour > 3.0 * level_direct_only


def test_door_motion_within_segment_smooth_and_block_independent():
    # Both doors move *inside* the rendered interval, with keyframes
    # landing at arbitrary sample positions relative to block boundaries.
    def build_scene():
        scene = _three_rooms()
        scene.portals[0].motion = DoorMotion(
            [(0.0, 1.0), (0.23, 1.0), (0.71, 0.0), (1.1, 0.0), (1.4, 1.0)]
        )
        scene.portals[1].motion = DoorMotion(
            [(0.0, 0.0), (0.55, 0.0), (0.9, 1.0)]
        )
        return scene

    def run(bs):
        listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
        eng = SpatialEngine(listener, sample_rate=SR, scene=_scene_copy(build_scene()))
        eng.add_emitter(
            Emitter("src", _tone(440.0, 2.0), position=(7.0, 0.0, 0.0))
        )
        return eng.render(1.6, block_size=bs).samples

    ref = run(48000)
    assert np.max(np.abs(np.diff(ref, axis=0))) < 0.05
    for bs in (17, 257, 1000):
        np.testing.assert_allclose(run(bs), ref, atol=1e-10)


def test_listener_crosses_boundary_at_block_edge_no_click():
    # Listener reaches the x=0 wall exactly at t=1 s (48000 samples).
    def build():
        listener = Listener(
            forward=(1.0, 0.0, 0.0),
            trajectory=Trajectory.linear(
                [(0.0, (-2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))]
            ),
        )
        eng = SpatialEngine(listener, sample_rate=SR,
                            scene=_three_rooms(bc_open=0.0))
        e = Emitter("src", _tone(440.0, 3.0), position=(2.5, 0.0, 0.0))
        eng.add_emitter(e)
        # Source stops before the crossing: its room-b reverb tail must
        # still ring across the boundary instead of being erased.
        eng.schedule_active(e, 0.8, False)
        return eng

    ref = build().render(1.6, block_size=48000).samples
    assert np.all(np.isfinite(ref))
    assert np.max(np.abs(np.diff(ref, axis=0))) < 0.05
    for bs in (96, 1000, 48000):
        np.testing.assert_allclose(build().render(1.6, block_size=bs).samples,
                                   ref, atol=1e-10)
    t = np.arange(len(ref)) / SR
    after_cross = ref[(t > 1.05) & (t < 1.3)]
    assert np.sqrt(np.mean(after_cross ** 2)) > 1e-4  # tail survives


def test_resume_mid_render_matches_continuous(tmp_path):
    scene = _three_rooms()
    scene.portals[0].motion = DoorMotion(
        [(0.0, 1.0), (0.3, 0.2), (1.1, 0.9)]
    )
    rec = MotionRecording(
        duration=2.0,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear(
            [(0.0, (-2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))]
        ),
        listener_forward=(1.0, 0.0, 0.0),
        scene=scene,
        emitters=[
            EmitterSpec(
                name="src",
                kind="tone",
                params={"frequency": 520.0, "duration": 2.0, "fade": 0.005},
                trajectory=Trajectory.linear([(0.0, (7.0, 0.0, 0.0))]),
                events=[(1.2, False)],
            )
        ],
    )
    ref = render_recording(rec, block_size=1024).samples

    # First half with one block size, suspend to disk, rebuild the engine,
    # restore, and finish with a different block size.
    engine = build_engine(rec)
    first = engine.render(0.731, block_size=257).samples
    state_path = tmp_path / "state.json"
    engine.save_state(state_path)
    second = render_recording_resumed(
        rec, json.loads(state_path.read_text()), block_size=333
    ).samples
    glued = np.concatenate([first, second], axis=0)
    assert glued.shape == ref.shape
    # The glue falls on the room crossing, where the signal moves
    # naturally: the step there must match the continuous render exactly.
    glue_step = np.abs(glued[len(first)] - glued[len(first) - 1])
    ref_step = np.abs(ref[len(first)] - ref[len(first) - 1])
    assert np.max(np.abs(glue_step - ref_step)) < 1e-6
    np.testing.assert_allclose(glued, ref, atol=1e-10)

    # Advancing a fresh engine first and then restoring the saved state
    # must produce the identical second half.
    engine2 = build_engine(rec)
    engine2.render(0.5, block_size=4096)
    engine2.load_state(state_path)
    rest = engine2.render(2.0 - 0.731, block_size=64).samples
    np.testing.assert_allclose(rest, ref[len(first):], atol=1e-10)


def test_replay_different_doors_keeps_source_schedule():
    scene = _three_rooms()
    rec = MotionRecording(
        duration=2.0,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear(
            [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))]
        ),
        listener_forward=(1.0, 0.0, 0.0),
        scene=scene,
        emitters=[
            EmitterSpec(
                name="src",
                kind="tone",
                params={"frequency": 520.0, "duration": 2.0, "fade": 0.005},
                trajectory=Trajectory.linear([(0.0, (7.0, 0.0, 0.0))]),
                events=[(0.5, True), (1.3, False)],
            )
        ],
    )
    alt = recording_with_doors(rec, {
        "ab": DoorMotion([(0.0, 0.0)]),
        "bc": DoorMotion([(0.0, 0.0)]),
    })
    # Original take's events are untouched; the copy keeps the same schedule.
    assert rec.emitters[0].events == [(0.5, True), (1.3, False)]
    assert alt.emitters is not rec.emitters
    assert alt.emitters[0].events == [(0.5, True), (1.3, False)]

    open_out = render_recording(rec, block_size=512).samples
    closed_out = render_recording(alt, block_size=512).samples
    assert open_out.shape == closed_out.shape
    t = np.arange(len(open_out)) / SR
    on = (t > 0.7) & (t < 1.2)
    e_open = np.sqrt(np.mean(open_out[on] ** 2))
    e_closed = np.sqrt(np.mean(closed_out[on] ** 2))
    assert e_open > 5.0 * e_closed
