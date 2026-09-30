"""Occlusion, doors (portals) and room reverb tests."""

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
    Waypoint,
    render_recording,
)
from spatial_audio.recording import EmitterSpec

SR = 48000


def _tone(freq=440.0, duration=4.0):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.005)


def _two_room_scene(door_keys):
    """Room A (listener) left of x=0 wall, room B (source) right of it."""
    rooms = [
        Room("a", (-5.0, -5.0, -2.0), (0.0, 5.0, 2.0), reverb_time=0.5),
        Room("b", (0.0, -5.0, -2.0), (5.0, 5.0, 2.0), reverb_time=0.5),
    ]
    portals = [Portal("door", "a", "b", DoorMotion(door_keys))]
    return RoomScene(rooms, portals)


def _engine_with_door(door_keys, duration=4.0, freq=440.0):
    listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    scene = _two_room_scene(door_keys)
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
    engine.add_emitter(
        Emitter("src", _tone(freq, duration), position=(2.0, 0.0, 0.0))
    )
    return engine


def test_door_close_open_block_size_independent():
    keys = [(0.0, 1.0), (1.0, 0.0), (2.0, 0.0), (3.0, 1.0)]
    ref = _engine_with_door(keys).render(3.5, block_size=48000).samples
    for bs in (64, 333, 1024, 5000):
        out = _engine_with_door(keys).render(3.5, block_size=bs).samples
        np.testing.assert_allclose(out, ref, atol=1e-10)


def test_door_motion_is_continuous_in_output():
    # Door swings closed over 0.5 s: output must have no sample-to-sample
    # jumps (no clicks from the changing gain/spectrum).
    keys = [(0.0, 1.0), (1.0, 1.0), (1.5, 0.0)]
    s = _engine_with_door(keys).render(2.5, block_size=257).samples
    step = np.max(np.abs(np.diff(s, axis=0)))
    assert step < 0.05


def test_closed_door_blocks_direct_but_tail_remains():
    # Door closes at t=1; source stops at t=2 (smoothed). After the stop,
    # the direct path is gone but room B's reverb tail must still be
    # audible through the closed door for a while.
    keys = [(0.0, 1.0), (1.0, 0.0)]
    listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    scene = _two_room_scene(keys)
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
    e = Emitter("src", _tone(440.0, 4.0), position=(2.0, 0.0, 0.0))
    engine.add_emitter(e)
    engine.schedule_active(e, 2.0, False)
    s = engine.render(4.0, block_size=512).samples
    t = np.arange(len(s)) / SR

    def rms(mask):
        seg = s[mask]
        return float(np.sqrt(np.mean(seg ** 2)))

    open_level = rms((t > 0.4) & (t < 0.9))
    closed_level = rms((t > 1.3) & (t < 1.9))
    tail_level = rms((t > 2.2) & (t < 2.6))
    silent_level = rms(t > 3.7)
    assert closed_level < 0.5 * open_level          # direct path blocked
    assert tail_level > 1e-4                        # reverb tail audible
    assert tail_level < closed_level                # and decaying
    assert silent_level < 1e-4                      # tail fully decayed


def test_closed_door_muffles_spectrum():
    def high_freq_energy(keys, t0, t1):
        s = _engine_with_door(keys, freq=3000.0).render(2.0, block_size=1024).samples
        seg = s[int(t0 * SR):int(t1 * SR), 0]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
        return float(spec[freqs > 2000.0].sum())

    open_hf = high_freq_energy([(0.0, 1.0)], 1.0, 1.5)
    closed_hf = high_freq_energy([(0.0, 0.0)], 1.0, 1.5)
    assert closed_hf < 0.3 * open_hf


def test_teleporting_source_no_click():
    listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    scene = _two_room_scene([(0.0, 1.0)])
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
    # Source jumps instantly from room B to far inside room A at t = 1.
    traj = Trajectory(
        [
            Waypoint(0.0, (2.0, 0.0, 0.0)),
            Waypoint(1.0, (2.0, 0.0, 0.0)),
            Waypoint(1.0 + 1e-9, (-4.0, 3.0, 0.0)),
            Waypoint(2.0, (-4.0, 3.0, 0.0)),
        ]
    )
    engine.add_emitter(Emitter("tp", _tone(500.0, 3.0), traj))
    s = engine.render(2.0, block_size=384).samples
    assert np.all(np.isfinite(s))
    step = np.max(np.abs(np.diff(s, axis=0)))
    assert step < 0.1  # smoothed gain change, no discontinuity


def test_listener_crossing_room_boundary_continuous():
    # Listener walks from room A through the doorway into room B.
    listener = Listener(
        forward=(1.0, 0.0, 0.0),
        trajectory=Trajectory.linear(
            [(0.0, (-2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))]
        ),
    )
    scene = _two_room_scene([(0.0, 1.0)])
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
    engine.add_emitter(Emitter("src", _tone(440.0, 3.0), position=(3.0, 1.0, 0.0)))
    ref = engine.render(2.0, block_size=48000).samples
    assert np.all(np.isfinite(ref))
    assert np.max(np.abs(np.diff(ref, axis=0))) < 0.05
    for bs in (96, 777, 4096):
        engine2 = SpatialEngine(
            Listener(
                forward=(1.0, 0.0, 0.0),
                trajectory=Trajectory.linear(
                    [(0.0, (-2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))]
                ),
            ),
            sample_rate=SR,
            scene=_two_room_scene([(0.0, 1.0)]),
        )
        engine2.add_emitter(
            Emitter("src", _tone(440.0, 3.0), position=(3.0, 1.0, 0.0))
        )
        np.testing.assert_allclose(
            engine2.render(2.0, block_size=bs).samples, ref, atol=1e-10
        )


def test_loud_mix_with_reverb_no_clipping_and_tail_kept():
    listener = Listener(position=(-2.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    scene = _two_room_scene([(0.0, 1.0)])
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene,
                           limit_threshold=0.95)
    emitters = []
    for i in range(8):
        e = Emitter(f"e{i}", _tone(300.0 + 40.0 * i, 2.0),
                    position=(-1.0, 1.0 + 0.3 * i, 0.0), volume=2.0)
        engine.add_emitter(e)
        engine.schedule_active(e, 1.5, False)
        emitters.append(e)
    r = engine.render(3.0, block_size=640)
    assert r.stats.pre_limiter_peak > 1.0
    assert r.stats.limited
    assert r.peak() <= 0.95 + 1e-9
    # Tail after all sources stopped: still nonzero (reverb rings on).
    t = np.arange(len(r.samples)) / SR
    tail = r.samples[(t > 1.7) & (t < 2.0)]
    assert np.sqrt(np.mean(tail ** 2)) > 1e-4


def test_toggling_one_emitter_does_not_move_others():
    # Mix is linear below the limiter: (A + B toggled) - (A alone) must
    # equal B rendered alone with the same toggle. B's start/stop cannot
    # shift A's playback position or phase.
    def render(with_b, toggle):
        listener = Listener(position=(0.0, 0.0, 0.0))
        engine = SpatialEngine(listener, sample_rate=SR)
        a = Emitter("a", _tone(440.0, 2.0), position=(0.0, 2.0, 0.0))
        engine.add_emitter(a)
        if with_b:
            b = Emitter("b", _tone(617.0, 2.0), position=(1.0, 3.0, 0.0))
            engine.add_emitter(b)
            if toggle:
                engine.schedule_active(b, 1.0, False)
        return engine.render(2.0, block_size=500).samples

    both = render(True, toggle=True)
    only_a = render(False, toggle=False)
    b_alone_engine = SpatialEngine(Listener(), sample_rate=SR)
    b = Emitter("b", _tone(617.0, 2.0), position=(1.0, 3.0, 0.0))
    b_alone_engine.add_emitter(b)
    b_alone_engine.schedule_active(b, 1.0, False)
    only_b = b_alone_engine.render(2.0, block_size=500).samples
    np.testing.assert_allclose(both, only_a + only_b, atol=1e-12)

    # Scheduling B's stop must not alter anything before the event.
    both_late = render(True, toggle=True)
    a_before = render(False, toggle=False)
    np.testing.assert_allclose(
        both_late[: SR] - a_before[: SR], only_b[: SR], atol=1e-12
    )


def test_recording_with_rooms_and_door_replay(tmp_path):
    scene = _two_room_scene([(0.0, 1.0), (1.0, 0.0), (2.0, 0.0), (2.5, 1.0)])
    rec = MotionRecording(
        duration=3.0,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear(
            [(0.0, (-2.0, 0.0, 0.0)), (3.0, (-1.0, 1.0, 0.0))]
        ),
        listener_forward=(1.0, 0.0, 0.0),
        scene=scene,
        emitters=[
            EmitterSpec(
                name="src",
                kind="tone",
                params={"frequency": 520.0, "duration": 3.0, "fade": 0.005},
                trajectory=Trajectory.linear([(0.0, (2.0, 0.0, 0.0))]),
                events=[(2.0, False)],
            )
        ],
    )
    path = tmp_path / "scene.json"
    rec.save(path)
    reloaded = MotionRecording.load(path)
    a = render_recording(rec, block_size=333).samples
    b = render_recording(reloaded, block_size=4096).samples
    np.testing.assert_allclose(a, b, atol=1e-12)
    # Door motion survived the round trip.
    door = reloaded.scene.portals[0].motion
    assert abs(door.open_fraction(0.0) - 1.0) < 1e-9
    assert abs(door.open_fraction(1.5) - 0.0) < 1e-9
    assert abs(door.open_fraction(3.0) - 1.0) < 1e-9
