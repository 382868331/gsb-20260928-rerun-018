import numpy as np

from spatial_audio import (
    Door,
    Emitter,
    Listener,
    MotionRecording,
    Occluder,
    Room,
    SoundResource,
    SpatialEngine,
    Trajectory,
    render_recording,
)
from spatial_audio.recording import EmitterSpec

SR = 48000


def _tone(freq=440.0, duration=3.0):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.005)


def _two_rooms():
    a = Room("A", (-5.0, -5.0), (0.0, 5.0), rt60=0.5)
    b = Room("B", (0.0, -5.0), (5.0, 5.0), rt60=0.7)
    return a, b


def _door_scene(open_events=(), block_move=False):
    a, b = _two_rooms()
    walls = [
        Occluder("w1", (0.0, -5.0), (0.0, -0.9)),
        Occluder("w2", (0.0, 0.9), (0.0, 5.0)),
    ]
    door = Door("door", (0.0, -0.9), (0.0, 0.9), openness=1.0)
    for t, o in open_events:
        door.schedule_openness(t, o)
    walls.append(door)
    if block_move:
        # Emitter stays in room A; listener crosses the wall (blocked view).
        src = Trajectory.linear([(0.0, (-3.0, 0.0, 0.0))])
        listener_traj = Trajectory.linear(
            [(0.0, (3.0, 0.0, 0.0)), (2.0, (-3.0, 0.0, 0.0))]
        )
        listener = Listener(trajectory=listener_traj)
    else:
        src = Trajectory.linear(
            [(0.0, (-3.0, 0.0, 0.0)), (2.0, (3.0, 0.0, 0.0))]
        )
        listener = Listener(position=(3.0, 0.0, 0.0))
    engine = SpatialEngine(
        listener, sample_rate=SR, rooms=[a, b], occluders=walls,
    )
    engine.add_emitter(Emitter("walker", _tone(440, 3.0), src))
    return engine, door


def test_doorway_passage_is_continuous():
    # Emitter walks straight through the open doorway: no click and the
    # level changes smoothly through the crossing (x = 0 around t = 1 s).
    engine, _ = _door_scene()
    s = engine.render(2.0, block_size=512).samples
    t = np.arange(len(s)) / SR
    region = (t > 0.94) & (t < 1.06)
    # Compare to the smooth carrier: the envelope must not introduce jumps
    # beyond what the 440 Hz sine itself produces. Measure RMS over short
    # windows and require a small per-window change.
    sig = s[region, 0]
    hop = SR // 200  # 5 ms
    frames = sig[: len(sig) // hop * hop].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    assert np.max(np.abs(np.diff(rms))) < 0.02


def test_block_size_independent_with_rooms_and_doors():
    ref = _door_scene(open_events=((0.8, 0.0), (1.4, 1.0)))[0].render(
        2.0, block_size=8192
    ).samples
    # 17 is far below the 128-sample filter window and the ~1.6 ms comb
    # delays, so it still crosses every internal boundary; block_size=1 is
    # covered for the (room-less) core engine in test_engine.
    for bs in (17, 333, 1024, 4097):
        out = _door_scene(open_events=((0.8, 0.0), (1.4, 1.0)))[0].render(
            2.0, block_size=bs
        ).samples
        np.testing.assert_allclose(out, ref, atol=1e-6, rtol=0)


def test_closed_door_blocks_direct_but_room_tail_survives():
    # Door closes at t = 0.6 s. The direct path (source in A, listener in
    # B) is rapidly attenuated, but the room reverb keeps an audible tail
    # that decays on its own RT60 rather than being cut off.
    a, b = _two_rooms()
    walls = [
        Occluder("w1", (0.0, -5.0), (0.0, -0.9)),
        Occluder("w2", (0.0, 0.9), (0.0, 5.0)),
        Door("door", (0.0, -0.9), (0.0, 0.9), openness=1.0),
    ]
    door = walls[-1]
    door.schedule_openness(0.4, 0.0)
    listener = Listener(position=(3.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR, rooms=[a, b], occluders=walls)
    emitter = Emitter(
        "static", _tone(440, 3.0),
        Trajectory.linear([(0.0, (-3.0, 0.0, 0.0))]),
    )
    engine.add_emitter(emitter)
    engine.schedule_active(emitter, 0.9, False)
    s = engine.render(2.2, block_size=1024).samples[:, 0]
    t = np.arange(len(s)) / SR
    # Energy just before closing vs shortly after (direct path blocked).
    before = np.sqrt(np.mean(s[(t > 0.3) & (t < 0.38)] ** 2))
    after = np.sqrt(np.mean(s[(t > 0.6) & (t < 0.75)] ** 2))
    assert after < before * 0.6
    # Tail is still present after the close and decays smoothly (no drop to
    # zero): RT60 of 0.7 s keeps energy alive well past the closing ramp.
    # After the source stops (0.9 s) the room keeps audibly ringing, then
    # decays following room B's RT60 rather than cutting instantly.
    early_tail = np.sqrt(np.mean(s[(t > 1.0) & (t < 1.1)] ** 2))
    late_tail = np.sqrt(np.mean(s[(t > 1.4) & (t < 1.5)] ** 2))
    assert early_tail > 1e-3
    assert late_tail < early_tail
    assert late_tail > early_tail * 10.0 ** (-3.0 * 0.4 / 0.7) * 0.1


def test_door_close_has_no_click():
    engine, _ = _door_scene(open_events=((0.5, 0.0),))
    s = engine.render(1.2, block_size=257).samples
    t = np.arange(len(s)) / SR
    around = (t > 0.48) & (t < 0.62)
    jumps = np.abs(np.diff(s[around, 0]))
    assert np.max(jumps) < 0.02


def test_start_stop_does_not_move_other_playheads():
    listener = Listener(position=(0.0, 3.0, 0.0))
    traj = Trajectory.linear([(0.0, (0.0, 2.0, 0.0))])

    def build(stop_b):
        engine = SpatialEngine(listener, sample_rate=SR)
        a = Emitter("a", _tone(440, 3.0), traj)
        b = Emitter("b", _tone(660, 3.0),
                     Trajectory.linear([(0.0, (2.0, 2.0, 0.0))]))
        engine.add_emitter(a)
        engine.add_emitter(b)
        if stop_b:
            engine.schedule_active(b, 0.5, False)
        return engine

    # Source A's contribution must be identical whether B exists/plays or
    # not: compare an A+B scene against an A-only scene directly.
    both = build(False).render(1.2, block_size=512).samples
    only_a = SpatialEngine(listener, sample_rate=SR)
    only_a.add_emitter(Emitter("a", _tone(440, 3.0), traj))
    ref = only_a.render(1.2, block_size=512).samples

    stopped = build(True).render(1.2, block_size=512).samples
    only_b = SpatialEngine(listener, sample_rate=SR)
    b_emit = Emitter("b", _tone(660, 3.0),
                     Trajectory.linear([(0.0, (2.0, 2.0, 0.0))]))
    only_b.add_emitter(b_emit)
    only_b.schedule_active(b_emit, 0.5, False)
    b_part = only_b.render(1.2, block_size=512).samples

    # Before B stops the mix is A + B (linearity); after it stops B has a
    # finite gate release, but A alone must remain exactly phase-aligned.
    n_early = int(0.45 * SR)
    np.testing.assert_allclose(
        both[:n_early] - b_part[:n_early], ref[:n_early], atol=1e-9
    )
    # Crucially A's waveform after B has fully released equals A-only.
    n_late0, n_late1 = int(0.8 * SR), int(1.2 * SR)
    np.testing.assert_allclose(
        stopped[n_late0:n_late1], ref[n_late0:n_late1], atol=1e-9
    )


def test_instant_emitter_teleport_has_no_pop():
    listener = Listener(position=(0.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR, smooth_ms=3.0)
    # Discontinuous trajectory: jump from far left to far right mid-take.
    traj = Trajectory.linear([
        (0.0, (-10.0, 2.0, 0.0)),
        (0.999, (-10.0, 2.0, 0.0)),
        (1.001, (10.0, 2.0, 0.0)),
        (2.0, (10.0, 2.0, 0.0)),
    ])
    engine.add_emitter(Emitter("jump", _tone(440, 3.0), traj))
    s = engine.render(2.0, block_size=333).samples
    t = np.arange(len(s)) / SR
    around = (t > 0.98) & (t < 1.05)
    assert np.max(np.abs(np.diff(s[around, 0]))) < 0.05


def test_listener_room_boundary_crossing_is_smooth():
    a, b = _two_rooms()
    listener_traj = Trajectory.linear(
        [(0.0, (-3.0, 0.0, 0.0)), (2.0, (3.0, 0.0, 0.0))]
    )
    listener = Listener(trajectory=listener_traj)
    engine = SpatialEngine(listener, sample_rate=SR, rooms=[a, b])
    engine.add_emitter(
        Emitter("loop", _tone(300, 3.0),
                Trajectory.linear([(0.0, (-3.0, 0.0, 0.0))]), loop=True)
    )
    s = engine.render(2.0, block_size=500).samples
    t = np.arange(len(s)) / SR
    around = (t > 0.94) & (t < 1.06)
    assert np.max(np.abs(np.diff(s[around, 0]))) < 0.02


def test_many_sources_peak_limited():
    listener = Listener()
    engine = SpatialEngine(listener, sample_rate=SR, limit_threshold=0.95)
    for i in range(24):
        engine.add_emitter(
            Emitter(f"e{i}", _tone(440 + i, 2.0), volume=2.0,
                    position=(0.0, 1.5, 0.0))
        )
    r = engine.render(1.0)
    assert r.stats.pre_limiter_peak > 1.0
    assert r.peak() <= 0.95 + 1e-9


def test_recording_persists_rooms_walls_and_door_events(tmp_path):
    a, b = _two_rooms()
    wall = Occluder("wall", (0.0, -5.0), (0.0, -1.0))
    door = Door("door", (0.0, -1.0), (0.0, 1.0), openness=1.0)
    door.schedule_openness(0.5, 0.0)
    rec = MotionRecording(
        duration=1.5,
        sample_rate=SR,
        rooms=[a, b],
        occluders=[wall, door],
        emitters=[
            EmitterSpec(
                name="beep",
                kind="tone",
                params={"frequency": 440.0, "duration": 1.5, "fade": 0.005},
                trajectory=Trajectory.linear(
                    [(0.0, (-3.0, 0.0, 0.0)), (1.5, (3.0, 0.0, 0.0))]
                ),
            )
        ],
    )
    path = tmp_path / "room_scene.json"
    rec.save(path)
    loaded = MotionRecording.load(path)
    assert [r.name for r in loaded.rooms] == ["A", "B"]
    kinds = [type(w).__name__ for w in loaded.occluders]
    assert kinds == ["Occluder", "Door"]
    assert loaded.occluders[1].events == [(0.5, 0.0)]

    a_out = render_recording(rec, block_size=333).samples
    b_out = render_recording(loaded, block_size=4099).samples
    np.testing.assert_allclose(a_out, b_out, atol=1e-9)
