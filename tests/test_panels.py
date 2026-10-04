"""Moving and revisable absorption panels.

A panel (AbsorptionZone) can slide/rotate over time (PanelMotion) and can
be revised for an explicitly selected propagation interval. Revisions only
recolor sound generated inside the interval; tails already emitted keep
the panel version they were produced with. Rendering with the revision
recorded from t=0 recomputes history from the boundary under the new
version, so both listening variants can be compared.
"""

import numpy as np

from spatial_audio import (
    AbsorptionZone,
    DoorMotion,
    Emitter,
    Listener,
    PanelMotion,
    Portal,
    Room,
    RoomScene,
    SoundResource,
    SpatialEngine,
)
from spatial_audio.rooms import RoomScene as _RoomScene

SR = 48000
LOW, MID, HIGH = 200.0, 1000.0, 4000.0
BAND_EDGES = (400.0, 2000.0)

LISTENER = (-2.0, 0.0, 0.0)
SOURCE = (4.0, 0.0, 0.0)

# Panel far along the direct path: crosses the direct path and the
# source -> room-center reverb send path, but never the detour.
FAR_MIN, FAR_MAX = (2.8, -0.6, -1.2), (3.2, 0.6, 1.2)
# Panel near the door: crosses only the direct path, not the send path.
NEAR_MIN, NEAR_MAX = (0.8, -0.6, -1.2), (1.2, 0.6, 1.2)

MOVE_KEYS = [(0.0, (3.0, 0.0, 0.0), 0.0),
             (3.0, (3.0, 0.0, 0.0), 0.0),
             (4.0, (3.0, 3.0, 0.0), 0.0)]


def _tone(freq, duration):
    return SoundResource.sine_tone(freq, duration, SR, fade=0.005)


def _rms(samples, t0, t1):
    seg = samples[int(t0 * SR):int(t1 * SR)]
    return float(np.sqrt(np.mean(seg ** 2)))


def _band_energy(samples, t0, t1, freq):
    seg = samples[int(t0 * SR):int(t1 * SR), 0]
    window = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * window)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
    k = int(np.argmin(np.abs(freqs - freq)))
    return float(spec[max(0, k - 3):k + 4].sum())


def _scene(zones=(), direct_door=True, detour=True):
    """Listener room a, source room b, detour room c (a-c-b)."""
    rooms = [
        Room("a", (-5.0, -5.0, -2.0), (0.0, 5.0, 2.0), reverb_time=0.8),
        Room("b", (0.0, -5.0, -2.0), (5.0, 5.0, 2.0), reverb_time=0.8),
        Room("c", (-5.0, 5.0, -2.0), (5.0, 10.0, 2.0), reverb_time=0.8),
    ]
    portals = []
    if direct_door:
        portals.append(Portal("ab", "a", "b", DoorMotion([(0.0, 1.0)]),
                              position=(0.0, 0.0, 0.0)))
    if detour:
        portals += [
            Portal("ac", "a", "c", DoorMotion([(0.0, 1.0)]),
                   position=(-2.0, 5.0, 0.0)),
            Portal("cb", "c", "b", DoorMotion([(0.0, 1.0)]),
                   position=(4.0, 5.0, 0.0)),
        ]
    return RoomScene(rooms, portals, zones, band_edges=BAND_EDGES)


def _render(scene, freq=HIGH, duration=6.0, events=(), block_size=512):
    listener = Listener(position=LISTENER, forward=(1.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR, scene=scene)
    emitter = Emitter("src", _tone(freq, duration), position=SOURCE)
    engine.add_emitter(emitter)
    for t, active in events:
        engine.schedule_active(emitter, t, active)
    return engine.render(duration, block_size=block_size).samples


def test_three_band_panel_is_band_selective():
    panel = AbsorptionZone("panel", FAR_MIN, FAR_MAX,
                           bands=[0.95, 0.4, 0.05])
    plain = _scene([], detour=False)
    treated = _scene([panel], detour=False)
    ratios = {}
    for name, freq in [("low", LOW), ("mid", MID), ("high", HIGH)]:
        a = _render(plain, freq=freq, duration=1.5)
        b = _render(treated, freq=freq, duration=1.5)
        ratios[name] = _rms(b, 0.5, 1.4) / _rms(a, 0.5, 1.4)
    assert ratios["low"] > 0.7
    assert ratios["high"] < 0.3
    assert ratios["low"] > ratios["mid"] > ratios["high"]


def test_detour_path_unaffected_by_moving_panel():
    # Only the detour a-c-b exists; the moving panel never crosses it,
    # so the render is bit-identical to the panel-free scene.
    panel = AbsorptionZone("panel", NEAR_MIN, NEAR_MAX,
                           bands=[0.95, 0.4, 0.05],
                           motion=[(0.0, (1.0, 0.0, 0.0), 0.0),
                                   (3.0, (1.0, 0.0, 0.0), 0.0),
                                   (4.0, (1.0, 3.0, 0.0), 0.0)])
    moved = _render(_scene([panel], direct_door=False), duration=5.0)
    plain = _render(_scene([], direct_door=False), duration=5.0)
    np.testing.assert_allclose(moved, plain, atol=1e-12, rtol=0)


def test_moving_panel_off_path_keeps_old_tail():
    # Panel crosses the direct path and the reverb send until t=3, then
    # slides away. The source stops at 2.5: the tail emitted while the
    # panel was in place must keep its historical (absorbed) spectrum.
    moving = AbsorptionZone("panel", FAR_MIN, FAR_MAX,
                            bands=[0.95, 0.4, 0.05], motion=MOVE_KEYS)
    static = AbsorptionZone("panel", FAR_MIN, FAR_MAX,
                            bands=[0.95, 0.4, 0.05])
    events = [(2.5, False)]
    out_moving = _render(_scene([moving]), events=events)
    out_static = _render(_scene([static]), events=events)
    out_plain = _render(_scene([]), events=events)
    # No new sound is generated after the move, so nothing changes.
    np.testing.assert_allclose(out_moving, out_static, atol=1e-9, rtol=0)
    # The old tail is still the absorbed one, not recomputed/cleared.
    e_treated = _band_energy(out_moving, 2.7, 3.4, HIGH)
    e_plain = _band_energy(out_plain, 2.7, 3.4, HIGH)
    assert e_treated < 0.5 * e_plain
    # Tail is alive right after the move, not cleared.
    assert _rms(out_moving, 2.7, 3.4) > 1e-5


def test_path_clear_after_move_for_new_sound():
    # After the panel slides away, newly generated sound takes the clear
    # direct path: much louder than with the panel still in place.
    moving = AbsorptionZone("panel", FAR_MIN, FAR_MAX,
                            bands=[0.95, 0.4, 0.05], motion=MOVE_KEYS)
    static = AbsorptionZone("panel", FAR_MIN, FAR_MAX,
                            bands=[0.95, 0.4, 0.05])
    events = [(2.5, False), (4.5, True)]
    out_moving = _render(_scene([moving]), events=events)
    out_static = _render(_scene([static]), events=events)
    out_plain = _render(_scene([]), events=events)
    level_moving = _rms(out_moving, 4.8, 5.9)
    assert level_moving > 1.5 * _rms(out_static, 4.8, 5.9)
    assert level_moving > 0.8 * _rms(out_plain, 4.8, 5.9)


def _revision_scenes():
    base_bands = [1.0, 1.0, 0.05]     # high band absorbed
    clear_bands = [1.0, 1.0, 0.9]     # revised: high band clear
    keep = AbsorptionZone("panel", FAR_MIN, FAR_MAX, bands=base_bands)
    keep.revise(2.0, 6.0, label="clear", bands=clear_bands)
    redo = AbsorptionZone("panel", FAR_MIN, FAR_MAX, bands=base_bands)
    # Recompute from the boundary: the revision reaches back to t=0, so
    # the tail decaying at the boundary is regenerated with the new
    # material instead of keeping the historical one.
    redo.revise(0.0, 6.0, label="clear", bands=clear_bands)
    norev = AbsorptionZone("panel", FAR_MIN, FAR_MAX, bands=base_bands)
    return keep, redo, norev


def test_revision_interval_keeps_history_unless_recomputed():
    keep, redo, norev = _revision_scenes()
    events = [(1.5, False), (2.5, True)]  # tail decays across the boundary
    # Single path here, so the revision factor is not diluted by the
    # (always clear) detour; the two-path case is covered above.
    out_keep = _render(_scene([keep], detour=False), events=events)
    out_redo = _render(_scene([redo], detour=False), events=events)
    out_norev = _render(_scene([norev], detour=False), events=events)

    # Before the revised interval the revision changes nothing.
    np.testing.assert_allclose(out_keep[:int(1.4 * SR)],
                               out_norev[:int(1.4 * SR)], atol=1e-9, rtol=0)
    # Tail decaying across the boundary keeps the historical material...
    np.testing.assert_allclose(out_keep[int(1.7 * SR):int(2.4 * SR)],
                               out_norev[int(1.7 * SR):int(2.4 * SR)],
                               atol=1e-9, rtol=0)
    # ...unless history is recomputed from the boundary: the same tail
    # then carries the revised (clear) high band.
    e_keep = _band_energy(out_keep, 1.8, 2.4, HIGH)
    e_redo = _band_energy(out_redo, 1.8, 2.4, HIGH)
    assert e_redo > 2.0 * e_keep
    # Sound generated inside the interval uses the revised version.
    assert _band_energy(out_keep, 3.0, 5.0, HIGH) > 2.0 * \
        _band_energy(out_norev, 3.0, 5.0, HIGH)


def test_version_segments_and_explain_report():
    zone = AbsorptionZone("panel", FAR_MIN, FAR_MAX, bands=[1.0, 1.0, 0.05])
    zone.revise(2.0, 4.0, label="clear", bands=[1.0, 1.0, 0.9])
    zone.revise(4.0, 6.0, label="moved",
                motion=[(0.0, (3.0, 3.0, 0.0), 0.0)])
    segments = zone.version_segments(0.0, 6.0)
    assert [s[2] for s in segments] == ["base", "clear", "moved"]
    assert (segments[0][0], segments[-1][1]) == (0.0, 6.0)

    scene = _scene([zone])
    info = scene.explain_path(LISTENER, SOURCE, 1.0)
    assert info["zones"][0]["version"] == "base"
    info = scene.explain_path(LISTENER, SOURCE, 3.0)
    assert info["zones"][0]["version"] == "clear"
    assert info["zones"][0]["bands"][2] == 0.9
    # After the panel moved away the path no longer crosses it.
    info = scene.explain_path(LISTENER, SOURCE, 5.0)
    assert info["zones"] == []


def _busy_scene():
    zone = AbsorptionZone("panel", FAR_MIN, FAR_MAX,
                          bands=[0.95, 0.4, 0.05], motion=MOVE_KEYS)
    zone.revise(1.0, 2.0, label="clear", bands=[0.95, 0.7, 0.9])
    return _scene([zone])


def test_block_size_independence_with_moving_revised_panel():
    events = [(0.8, False), (1.2, True)]
    ref = _render(_busy_scene(), duration=1.5, events=events, block_size=1024)
    for bs in (1, 37, 256, 10007):
        out = _render(_busy_scene(), duration=1.5, events=events, block_size=bs)
        np.testing.assert_allclose(out, ref, atol=1e-10, rtol=0)


def test_render_interval_matches_full_render():
    events = [(0.8, False), (1.2, True)]
    full = _render(_busy_scene(), duration=2.0, events=events)

    listener = Listener(position=LISTENER, forward=(1.0, 0.0, 0.0))
    engine = SpatialEngine(listener, sample_rate=SR, scene=_busy_scene())
    emitter = Emitter("src", _tone(HIGH, 2.0), position=SOURCE)
    engine.add_emitter(emitter)
    for t, active in events:
        engine.schedule_active(emitter, t, active)
    piece = engine.render_interval(0.5, 1.5, block_size=256).samples
    np.testing.assert_allclose(
        piece, full[int(0.5 * SR):int(1.5 * SR)], atol=1e-12, rtol=0)


def test_scene_dict_roundtrip_keeps_motion_and_revisions():
    scene = _busy_scene()
    clone = _RoomScene.from_dict(scene.to_dict())
    zone = clone.zones[0]
    assert zone.has_motion and zone.motion is not None
    assert [r["label"] for r in zone.revisions] == ["clear"]
    events = [(0.8, False)]
    a = _render(scene, duration=1.5, events=events)
    b = _render(clone, duration=1.5, events=events)
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
