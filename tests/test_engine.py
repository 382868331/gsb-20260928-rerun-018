import numpy as np

from spatial_audio import (
    LinearPath,
    Listener,
    SineTone,
    SoundSource,
    SpatialEngine,
    Vec3,
)


def _engine(sources):
    return SpatialEngine(sample_rate=48000, listener=Listener(Vec3(0, 0, 0)), sources=sources)


def render_chunked(sources, duration, chunk):
    eng = _engine(sources)
    n = int(duration * eng.sample_rate)
    parts = []
    pos = 0
    while pos < n:
        cur = min(chunk, n - pos)
        parts.append(eng.process(cur))
        pos += cur
    return np.concatenate(parts), eng


def test_chunk_size_invariance():
    def make():
        return [SoundSource(
            signal=SineTone(440, 0.5),
            trajectory=LinearPath([(0, -5, 3, 0), (4, 5, 3, 0)]),
        )]
    a, _ = render_chunked(make(), 4.0, 1)
    b, _ = render_chunked(make(), 4.0, 37)
    c, _ = render_chunked(make(), 4.0, 10000)
    assert np.max(np.abs(a - b)) < 1e-12
    assert np.max(np.abs(a - c)) < 1e-12


def test_left_to_right_pan():
    src = SoundSource(
        signal=SineTone(300, 0.5),
        trajectory=LinearPath([(0, -6, 3, 0), (4, 6, 3, 0)]),
    )
    audio, _ = render_chunked([src], 4.0, 1024)
    sr = 48000
    win = sr  # 1 s windows
    left_first = np.mean(audio[:win, 0] ** 2)
    right_first = np.mean(audio[:win, 1] ** 2)
    left_last = np.mean(audio[-win:, 0] ** 2)
    right_last = np.mean(audio[-win:, 1] ** 2)
    assert left_first > right_first * 1.5
    assert right_last > left_last * 1.5


def test_distance_attenuation():
    def render_at(x):
        src = SoundSource(signal=SineTone(200, 1.0), position=Vec3(0, x, 0))
        eng = _engine([src])
        full = eng.process(int(2.0 * 48000))
        arrival = int((x / 343.0) * 48000) + 100
        return full[arrival:arrival + int(0.1 * 48000)]  # steady state
    near = render_at(0.5)
    far = render_at(20.0)
    assert np.sqrt(np.mean(near ** 2)) > np.sqrt(np.mean(far ** 2)) * 3


def test_abrupt_deactivation_is_smooth_and_reaches_zero():
    src = SoundSource(signal=SineTone(440, 0.8), position=Vec3(0, 2, 0), deactivate_at=0.5)
    audio, eng = render_chunked([src], 1.0, 64)
    sr = 48000
    # Deactivation at emission t=0.5s reaches the listener after propagation.
    stop_out = int((0.5 + 2.0 / 343.0) * sr)
    tail = audio[stop_out:stop_out + int(0.03 * sr)]
    max_step = np.max(np.abs(np.diff(tail[:, 0])))
    # A hard on/off click at this amplitude would jump ~1.6; the smooth
    # release keeps the largest sample step two orders smaller.
    assert max_step < 0.25
    assert np.all(audio[stop_out + int(0.04 * sr):] == 0.0)
    assert eng.report.stopped_sources == ["source"]


def test_mixed_sources_cannot_silently_overflow():
    sources = [
        SoundSource(signal=SineTone(440, 1.0), position=Vec3(0, 2, 0))
        for _ in range(20)
    ]
    audio, eng = render_chunked(sources, 0.5, 512)
    assert np.all(np.abs(audio) <= 1.0)
    assert eng.report.over_level_events > 0
    assert eng.report.peak_pre_limiter > 1.0


def test_listener_and_source_update_independently():
    listener = Listener(Vec3(0, 0, 0))
    src = SoundSource(signal=SineTone(440, 0.5), position=Vec3(0, 0, 0), active=False)
    eng = SpatialEngine(sample_rate=48000, listener=listener, sources=[src])
    first = eng.process(2400)
    assert np.all(first == 0.0)
    # Activate the source, move the listener: independent state updates.
    src.update(active=True)
    listener.update(position=Vec3(0, 0, 0))
    src.update(position=Vec3(1, 1, 0))
    second = eng.process(2400)
    assert np.max(np.abs(second)) > 0.0


def test_doppler_changes_pitch_on_radial_motion():
    # Source flying fast past the listener; the delayed signal pitch shifts.
    src = SoundSource(
        signal=SineTone(1000, 0.6),
        trajectory=LinearPath([(0, -2, 0.3, 0), (1, 150, 0.3, 0)]),
    )
    audio, _ = render_chunked([src], 1.0, 2048)
    mono = audio[:, 0]
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / 48000)
    peak = freqs[int(np.argmax(spec))]
    assert peak != 1000.0  # motion shifted the observed pitch
