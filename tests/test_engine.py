import numpy as np
import pytest

from spatial_audio import (
    Emitter,
    Listener,
    MotionRecording,
    SoundResource,
    SpatialEngine,
    Trajectory,
    Waypoint,
    render_recording,
)

SR = 48000


def _tone(freq=440.0, duration=3.0):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.005)


def _moving_engine():
    listener = Listener(position=(0.0, 0.0, 0.0), forward=(0.0, 1.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR)
    traj = Trajectory(
        [
            Waypoint(0.0, (-8.0, 2.0, 0.0)),   # left of listener
            Waypoint(1.0, (0.0, 3.0, 0.0)),    # in front, closer
            Waypoint(2.0, (8.0, 2.0, 0.0)),    # right
        ]
    )
    engine.add_emitter(Emitter("moving", _tone(), traj))
    return engine


def test_block_size_independence():
    ref = _moving_engine().render(2.5, block_size=48000).samples
    for bs in (1, 37, 256, 1024, 10007):
        out = _moving_engine().render(2.5, block_size=bs).samples
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-10, rtol=0)


def test_left_right_pan_changes():
    result = _moving_engine().render(2.5, block_size=512)
    s = result.samples
    # Starts on the left: left channel louder early; ends louder on the right.
    early = slice(0, SR // 4)
    late = slice(-SR // 4, None)
    assert np.abs(s[early, 0]).mean() > np.abs(s[early, 1]).mean()
    assert np.abs(s[late, 1]).mean() > np.abs(s[late, 0]).mean()
    # Both channels are actually modulated.
    assert s.std() > 0


def test_distance_attenuation():
    listener = Listener()
    engine = SpatialEngine(listener, sample_rate=SR)
    near = Emitter("near", _tone(220.0, 2.0), position=(0.0, 1.0, 0.0))
    far = Emitter("far", _tone(220.0, 2.0), position=(0.0, 20.0, 0.0))
    engine.add_emitter(near)
    engine.add_emitter(far)
    r = engine.render(1.0)
    assert r.stats.emitter_peaks["near"] > r.stats.emitter_peaks["far"] * 5


def test_sudden_deactivation_is_smoothed():
    listener = Listener()
    engine = SpatialEngine(listener, sample_rate=SR)
    e = Emitter("e", _tone(440.0, 3.0), position=(0.0, 2.0, 0.0))
    engine.add_emitter(e)
    engine.schedule_active(e, 1.0, False)
    s = engine.render(2.0, block_size=257).samples
    t = np.arange(len(s)) / SR
    on = s[t < 0.95]
    assert np.max(np.abs(on)) > 0.05
    # After the off time, amplitude must decay monotonically (in short
    # windows) and reach near zero; no discontinuity/click at the event.
    tail = s[t >= 1.0]
    env = np.maximum.reduceat(
        np.abs(tail[:, 0]), np.arange(0, len(tail), SR // 100)[:-1]
    )
    assert np.all(np.diff(env) <= 1e-3)
    assert np.max(np.abs(s[t > 1.1])) < 1e-3


def test_deactivation_block_size_independence():
    def run(bs):
        listener = Listener()
        engine = SpatialEngine(listener, sample_rate=SR)
        e = Emitter("e", _tone(440.0, 3.0), position=(0.0, 2.0, 0.0))
        engine.add_emitter(e)
        engine.schedule_active(e, 0.7777, False)
        return engine.render(1.5, block_size=bs).samples

    np.testing.assert_allclose(run(1), run(64), atol=1e-10)
    np.testing.assert_allclose(run(64), run(48000), atol=1e-10)


def test_mixing_cannot_overflow():
    listener = Listener()
    engine = SpatialEngine(listener, sample_rate=SR, limit_threshold=0.95)
    for i in range(20):
        engine.add_emitter(
            Emitter(f"e{i}", _tone(440.0 + i, 2.0),
                    position=(0.0, 1.5, 0.0), volume=2.0)
        )
    r = engine.render(1.0)
    assert r.stats.pre_limiter_peak > 1.0  # raw mix would clip
    assert r.stats.limited is True
    assert r.peak() <= 0.95 + 1e-9


def test_doppler_changes_pitch():
    listener = Listener()
    engine = SpatialEngine(listener, sample_rate=SR)
    traj = Trajectory(
        [Waypoint(0.0, (-30.0, 0.0, 0.0)), Waypoint(2.0, (30.0, 0.0, 0.0))]
    )
    engine.add_emitter(Emitter("fast", _tone(400.0, 3.0), traj))
    s = engine.render(2.0, block_size=2048).samples[:, 0]

    def inst_freq(seg):
        sig = seg * np.hanning(len(seg))
        spec = np.abs(np.fft.rfft(sig))
        bin_index = int(np.argmax(spec[1:])) + 1
        return bin_index * SR / len(seg)

    f_approach = inst_freq(s[int(0.3 * SR):int(0.3 * SR) + SR // 2])
    f_recede = inst_freq(s[int(1.5 * SR):int(1.5 * SR) + SR // 2])
    assert f_approach > 410.0
    assert f_recede < 390.0
    assert f_approach > f_recede


def test_recording_replay_bit_identical(tmp_path):
    listener_traj = Trajectory(
        [Waypoint(0.0, (0.0, 0.0, 0.0)), Waypoint(3.0, (0.0, -2.0, 0.0))]
    )
    source_traj = Trajectory.linear(
        [(0.0, (-6.0, 2.0, 0.0)), (3.0, (6.0, 2.0, 0.0))]
    )
    rec = MotionRecording(
        duration=3.0,
        sample_rate=SR,
        listener_trajectory=listener_traj,
    )
    from spatial_audio.recording import EmitterSpec
    rec.emitters = [
        EmitterSpec(
            name="beep",
            kind="tone",
            params={"frequency": 500.0, "duration": 3.0, "fade": 0.005},
            trajectory=source_traj,
            events=[(1.0, True), (2.0, False)],
        )
    ]
    path = tmp_path / "motion.json"
    rec.save(path)

    reloaded = MotionRecording.load(path)
    a = render_recording(rec, block_size=333).samples
    b = render_recording(reloaded, block_size=4096).samples
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_resource_and_states_update_independently():
    listener = Listener(position=(0.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR)
    e = Emitter("e", _tone(440.0, 1.0), position=(0.0, 2.0, 0.0))
    engine.add_emitter(e)
    # Render, then swap resource and move listener independently; no error.
    engine.render(0.1)
    e.set_resource(_tone(880.0, 1.0))
    listener.set_position((1.0, 0.0, 0.0))
    e.set_active(False)
    r = engine.render(0.1)
    assert r.samples.shape[1] == 2
