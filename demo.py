"""Demo: artificial tones moving on trajectories -> stereo WAV + L/R check.

Run:
    .venv/bin/python demo.py

Outputs (in outputs/):
    demo_spatial.wav      audible stereo result (48 kHz, 16-bit PCM)
    motion_recording.json saved motion (trajectories + events), replayable
    channel_check.png     per-20 ms RMS of left/right channels to verify
                          the pan and distance changes
"""

from pathlib import Path

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from spatial_audio import (
    AbsorptionZone,
    DoorMotion,
    MotionRecording,
    Panel,
    PanelRevision,
    Portal,
    Room,
    RoomScene,
    Trajectory,
    load_checkpoint,
    panel_version_manifest,
    render_recording,
    save_checkpoint,
)
from spatial_audio.recording import EmitterSpec, build_engine

SR = 48000
DURATION = 6.0
OUT = Path("outputs")


def build_recording():
    # Main source: sweeps from far-left, through front-center (closest at
    # t = 3 s) to far-right.
    moving = Trajectory.linear(
        [
            (0.0, (-12.0, 6.0, 0.0)),
            (3.0, (0.0, 2.0, 0.0)),
            (6.0, (12.0, 6.0, 0.0)),
        ]
    )
    # Second source: stationary on the right; switched off suddenly at t = 2
    # (engine ramps it out smoothly).
    right_beep = Trajectory.linear([(0.0, (3.0, 2.0, 0.0))])

    return MotionRecording(
        duration=DURATION,
        sample_rate=SR,
        listener_trajectory=None,
        emitters=[
            EmitterSpec(
                name="moving_tone",
                kind="tone",
                params={"frequency": 440.0, "duration": DURATION, "fade": 0.01},
                trajectory=moving,
            ),
            EmitterSpec(
                name="right_beep",
                kind="tone",
                params={"frequency": 660.0, "duration": DURATION, "fade": 0.01},
                trajectory=right_beep,
                events=[(2.0, False)],
            ),
        ],
    )


def channel_rms(samples, hop_ms=20.0):
    hop = int(SR * hop_ms / 1000.0)
    n = len(samples) // hop
    frames = samples[: n * hop].reshape(n, hop, 2)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    t = np.arange(n) * hop_ms / 1000.0
    return t, rms


def main():
    OUT.mkdir(exist_ok=True)
    recording = build_recording()
    recording.save(OUT / "motion_recording.json")

    result = render_recording(recording, block_size=1024)
    wav_path = OUT / "demo_spatial.wav"
    sf.write(wav_path, result.samples, SR, subtype="PCM_16")

    # Replay from the saved recording with a different block size.
    replayed = MotionRecording.load(OUT / "motion_recording.json")
    replay_result = render_recording(replayed, block_size=4099)
    replay_path = OUT / "demo_spatial_replay.wav"
    sf.write(replay_path, replay_result.samples, SR, subtype="PCM_16")
    max_diff = float(np.max(np.abs(result.samples - replay_result.samples)))

    t, rms = channel_rms(result.samples)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, rms[:, 0], label="left", color="#1f77b4")
    axes[0].plot(t, rms[:, 1], label="right", color="#d62728")
    axes[0].set_ylabel("RMS level")
    axes[0].set_title("Spatial stereo output: L vs R over time")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    balance = (rms[:, 1] - rms[:, 0]) / np.maximum(rms.sum(axis=1), 1e-9)
    axes[1].plot(t, balance, color="black")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("R-L balance  (-1 left, +1 right)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "channel_check.png", dpi=130)

    st = result.stats
    print(f"wrote {wav_path} ({len(result.samples) / SR:.1f} s, {SR} Hz stereo)")
    print(f"saved motion: {OUT / 'motion_recording.json'}")
    print(f"replay max |difference| vs original: {max_diff:.3e}")
    print(f"per-emitter peaks: {st.emitter_peaks}")
    print(f"pre-limiter peak: {st.pre_limiter_peak:.3f}, "
          f"post-limiter peak: {st.post_limiter_peak:.3f}, "
          f"limiter engaged: {st.limited}")

    # Numerical sanity checks for the L/R sweep, evaluated after the
    # stationary right beep has been switched off, and away from its tail.
    early = slice(int(2.4 * SR), int(2.8 * SR))   # moving source still left
    late = slice(int(5.4 * SR), int(5.8 * SR))    # moving source now right
    assert np.abs(result.samples[early, 0]).mean() > np.abs(result.samples[early, 1]).mean()
    assert np.abs(result.samples[late, 1]).mean() > np.abs(result.samples[late, 0]).mean()
    assert max_diff < 1e-9
    print("L/R sweep and replay-equivalence checks passed.")

    demo_rooms()


def build_room_recording():
    """Two rooms joined by a door; the door closes, then reopens."""
    scene = RoomScene(
        rooms=[
            Room("hall", (-6.0, -4.0, -2.0), (0.0, 4.0, 2.0), reverb_time=0.7),
            Room("chamber", (0.0, -4.0, -2.0), (6.0, 4.0, 2.0), reverb_time=0.9),
        ],
        portals=[
            # Open until t=1.5, swings shut by t=2.0, stays closed, then
            # reopens between t=4.0 and t=4.5.
            Portal("door", "hall", "chamber",
                   DoorMotion([(0.0, 1.0), (1.5, 1.0), (2.0, 0.0),
                               (4.0, 0.0), (4.5, 1.0)])),
        ],
    )
    return MotionRecording(
        duration=DURATION,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear(
            [(0.0, (-3.0, 0.0, 0.0)), (6.0, (-1.0, 1.0, 0.0))]
        ),
        listener_forward=(1.0, 0.0, 0.0),
        scene=scene,
        emitters=[
            # Tone in the other room: direct sound is muffled and quiet
            # while the door is closed, but the chamber's reverb tail
            # remains audible after the tone stops at t=5.
            EmitterSpec(
                name="chamber_tone",
                kind="tone",
                params={"frequency": 520.0, "duration": DURATION, "fade": 0.01},
                trajectory=Trajectory.linear([(0.0, (2.5, 0.5, 0.0))]),
                events=[(5.0, False)],
            ),
            # A second tone that teleports inside the hall at t=3
            # (smoothed by the engine; no click).
            EmitterSpec(
                name="hall_tone",
                kind="tone",
                params={"frequency": 330.0, "duration": DURATION, "fade": 0.01},
                trajectory=Trajectory.linear(
                    [(0.0, (4.0, -2.0, 0.0)), (3.0, (4.0, -2.0, 0.0)),
                     (3.0 + 1e-9, (-4.0, 2.0, 0.0)), (6.0, (-4.0, 2.0, 0.0))]
                ),
            ),
        ],
    )


def demo_rooms():
    recording = build_room_recording()
    recording.save(OUT / "room_recording.json")

    result = render_recording(recording, block_size=1024)
    wav_path = OUT / "demo_rooms.wav"
    sf.write(wav_path, result.samples, SR, subtype="PCM_16")

    replayed = MotionRecording.load(OUT / "room_recording.json")
    replay = render_recording(replayed, block_size=4099).samples
    sf.write(OUT / "demo_rooms_replay.wav", replay, SR, subtype="PCM_16")
    max_diff = float(np.max(np.abs(result.samples - replay)))

    t, rms = channel_rms(result.samples)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, rms[:, 0], label="left", color="#1f77b4")
    ax.plot(t, rms[:, 1], label="right", color="#d62728")
    for x, label in [(2.0, "door closed"), (4.5, "door open"), (5.0, "tone off")]:
        ax.axvline(x, color="gray", ls="--", lw=0.8)
        ax.text(x, ax.get_ylim()[1] * 0.9, label, rotation=90, va="top", fontsize=8)
    ax.set_ylabel("RMS level")
    ax.set_xlabel("time (s)")
    ax.set_title("Two rooms + door: occlusion and reverb tail")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "room_check.png", dpi=130)

    print(f"wrote {wav_path} (door closes at 2 s, reopens at 4.5 s)")
    print(f"room replay max |difference|: {max_diff:.3e}")
    assert max_diff < 1e-9
    # Door closed (2.5-4 s) must be clearly quieter than open (1-1.5 s),
    # and the tail after the tone stops (5.2-5.6 s) must still be audible.
    def level(t0, t1):
        seg = result.samples[int(t0 * SR):int(t1 * SR)]
        return float(np.sqrt(np.mean(seg ** 2)))
    assert level(2.2, 2.9) < 0.6 * level(1.0, 1.5)
    assert level(5.2, 5.6) > 1e-4
    print("door occlusion and reverb-tail checks passed.")


CORRIDOR_DURATION = 8.0


def build_corridor_scene(door2_keys=None, with_treatment=False):
    """Corridor joined to two rooms; room1 <-> room2 is a detour route."""
    scene = RoomScene(
        rooms=[
            Room("corridor", (-2.0, -1.0, -2.0), (14.0, 1.0, 2.0),
                 reverb_time=0.5, reverb_send=0.2),
            Room("room1", (2.0, 1.0, -2.0), (7.0, 6.0, 2.0),
                 reverb_time=0.8, reverb_send=0.3),
            Room("room2", (8.0, 1.0, -2.0), (13.0, 6.0, 2.0),
                 reverb_time=0.9, reverb_send=0.3),
        ],
        portals=[
            Portal("door1", "corridor", "room1",
                   DoorMotion([(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]),
                   position=(4.5, 1.0, 0.0)),
            Portal("door2", "corridor", "room2",
                   DoorMotion(door2_keys if door2_keys is not None
                              else [(0.0, 0.0), (4.0, 0.0), (5.0, 1.0)]),
                   position=(10.5, 1.0, 0.0)),
            Portal("door3", "room1", "room2",
                   DoorMotion([(0.0, 1.0)]),
                   position=(7.5, 3.5, 0.0)),
        ],
    )
    if with_treatment:
        # Absorbing panels inside room1 around the door3 area: only paths
        # routed through that corner are damped.
        scene.zones.append(
            AbsorptionZone("panels", (6.5, 2.5, -2.0), (7.5, 4.5, 2.0),
                           gain=0.25, muffle=0.4)
        )
    return scene


def build_corridor_recording():
    """One walk take: the source travels the corridor past both rooms."""
    return MotionRecording(
        duration=CORRIDOR_DURATION,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear(
            [(0.0, (10.5, 3.0, 0.0)), (8.0, (11.5, 4.0, 0.0))]
        ),
        listener_forward=(-1.0, 0.0, 0.0),
        scene=build_corridor_scene(),
        emitters=[
            EmitterSpec(
                name="walker",
                kind="tone",
                params={"frequency": 500.0, "duration": CORRIDOR_DURATION,
                        "fade": 0.01},
                trajectory=Trajectory.linear(
                    [(0.0, (-1.0, 0.0, 0.0)), (7.0, (13.0, 0.0, 0.0))]
                ),
                events=[(6.5, False)],  # stops; tails keep decaying
            ),
        ],
    )


def _static_listener_recording(base, position):
    rec = MotionRecording(
        duration=base.duration,
        sample_rate=base.sample_rate,
        listener_trajectory=Trajectory.linear([(0.0, position)]),
        listener_forward=base.listener_forward,
        emitters=base.emitters,
        scene=base.scene,
    )
    return rec


def demo_corridor():
    recording = build_corridor_recording()
    recording.save(OUT / "corridor_recording.json")

    # Full take: source walks the corridor; door2 (corridor<->room2, where
    # the listener sits) stays shut until t=5, so early on the source is
    # heard through the detour corridor -> room1 -> room2 instead.
    result = render_recording(recording, block_size=1024)
    sf.write(OUT / "corridor_walk.wav", result.samples, SR, subtype="PCM_16")

    # Same walk take replayed under a different door arrangement.
    alt_scene = build_corridor_scene(door2_keys=[(0.0, 1.0)])
    alt_scene.portals[2].motion = DoorMotion([(0.0, 1.0), (3.0, 0.0)])
    alt = render_recording(recording, block_size=1024, scene=alt_scene)
    sf.write(OUT / "corridor_alt_doors.wav", alt.samples, SR, subtype="PCM_16")

    # Absorption retrofit comparison: same take, original walls vs added
    # panels in room1, compared at two listening positions.
    treated_scene = build_corridor_scene(with_treatment=True)
    spots = {
        "room2": (10.5, 3.0, 0.0),
        "corridor": (6.0, 0.0, 0.0),
    }
    curves = {}
    levels = {}
    for label, pos in spots.items():
        base_rec = _static_listener_recording(recording, pos)
        plain = render_recording(base_rec, block_size=1024)
        treated = render_recording(base_rec, block_size=1024,
                                   scene=treated_scene)
        sf.write(OUT / f"absorb_{label}_plain.wav", plain.samples, SR,
                 subtype="PCM_16")
        sf.write(OUT / f"absorb_{label}_treated.wav", treated.samples, SR,
                 subtype="PCM_16")
        curves[label] = (channel_rms(plain.samples), channel_rms(treated.samples))
        levels[label] = (plain.samples, treated.samples)

    # Save a checkpoint mid-take and resume listening later: the resumed
    # part must be identical to the uninterrupted render, including the
    # reverb tails still decaying at the checkpoint.
    engine = build_engine(recording)
    engine.render_chunk(4.0)
    save_checkpoint(recording, engine, OUT / "corridor_checkpoint.json")
    _, resumed_engine = load_checkpoint(OUT / "corridor_checkpoint.json")
    resumed = resumed_engine.render_chunk(CORRIDOR_DURATION - 4.0)
    sf.write(OUT / "corridor_resumed.wav", resumed.samples, SR,
             subtype="PCM_16")
    tail_diff = float(np.max(np.abs(
        resumed.samples - result.samples[int(4.0 * SR):]
    )))

    # A selected interval matches the same slice of the full render.
    interval_engine = build_engine(recording)
    interval = interval_engine.render_interval(2.0, 5.0, block_size=512)
    interval_diff = float(np.max(np.abs(
        interval.samples - result.samples[int(2.0 * SR):int(5.0 * SR)]
    )))

    t, rms = channel_rms(result.samples)
    _, rms_alt = channel_rms(alt.samples)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, rms[:, 0], label="original doors", color="#1f77b4")
    axes[0].plot(t, rms_alt[:, 0], label="alt door arrangement",
                 color="#d62728", alpha=0.8)
    for x, label in [(2.0, "door1 open"), (3.0, "door3 shut (alt)"),
                     (5.0, "door2 open"), (6.5, "source off")]:
        axes[0].axvline(x, color="gray", ls="--", lw=0.8)
        axes[0].text(x, axes[0].get_ylim()[1] * 0.9, label, rotation=90,
                     va="top", fontsize=8)
    axes[0].set_ylabel("RMS level")
    axes[0].set_title("Corridor walk heard from room2 (left channel)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for label, (plain_c, treated_c) in curves.items():
        axes[1].plot(plain_c[0], plain_c[1][:, 0],
                     label=f"{label} original")
        axes[1].plot(treated_c[0], treated_c[1][:, 0], ls="--",
                     label=f"{label} + panels")
    axes[1].axvline(6.5, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("RMS level")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Absorption retrofit: panels only damp paths through room1")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "corridor_check.png", dpi=130)

    print(f"wrote {OUT / 'corridor_walk.wav'} (corridor + 2 rooms, detour paths)")
    print(f"alt door arrangement -> {OUT / 'corridor_alt_doors.wav'}")
    print(f"absorption comparison -> absorb_*_plain/treated.wav, corridor_check.png")
    print(f"checkpoint resume max |difference|: {tail_diff:.3e}")
    print(f"interval vs full-slice max |difference|: {interval_diff:.3e}")
    assert tail_diff < 1e-9
    assert interval_diff < 1e-9
    # While door2 is shut (2.5-4 s) the detour keeps the source audible in
    # room2; after the source stops (6.5 s) the tails decay to silence.
    def level(samples, t0, t1):
        seg = samples[int(t0 * SR):int(t1 * SR)]
        return float(np.sqrt(np.mean(seg ** 2)))
    assert level(result.samples, 2.5, 3.5) > 1e-3
    assert level(result.samples, 6.6, 7.0) > 1e-5
    assert level(result.samples, 7.5, 8.0) < 0.01 * level(result.samples, 6.6, 7.0)
    # Panels damp the detour-heavy path into room2 while door2 is shut...
    plain_r2, treated_r2 = levels["room2"]
    assert level(treated_r2, 2.5, 3.5) < 0.8 * level(plain_r2, 2.5, 3.5)
    # ...but a corridor listener whose path avoids the panels is untouched.
    plain_c, treated_c = levels["corridor"]
    assert level(treated_c, 0.2, 0.8) > 0.8 * level(plain_c, 0.2, 0.8)
    print("corridor/detour/checkpoint/interval checks passed.")


def _band_split(samples, edge=1000.0):
    """Analysis helper: complementary one-pole low/high split."""
    a0 = 1.0 - np.exp(-2.0 * np.pi * edge / SR)
    from scipy.signal import lfilter
    low = np.empty_like(samples)
    for ch in range(2):
        low[:, ch] = lfilter([a0], [1.0, -(1.0 - a0)], samples[:, ch])
    return low, samples - low


def demo_bands():
    """Frequency-band absorption: same walk take, band-selective panels."""
    # Two walkers on the same corridor path: a 500 Hz tone (low band) and
    # a 4 kHz tone (high band), so each band can be checked separately.
    recording = build_corridor_recording()
    walk = Trajectory.linear(
        [(0.0, (-1.0, 0.0, 0.0)), (7.0, (13.0, 0.0, 0.0))]
    )
    recording.emitters = [
        EmitterSpec(
            name="walker_low",
            kind="tone",
            params={"frequency": 500.0, "duration": CORRIDOR_DURATION,
                    "fade": 0.01},
            trajectory=walk,
            events=[(6.5, False)],
        ),
        EmitterSpec(
            name="walker_high",
            kind="tone",
            params={"frequency": 4000.0, "duration": CORRIDOR_DURATION,
                    "fade": 0.01},
            trajectory=walk,
            events=[(6.5, False)],
        ),
    ]
    base_rec = _static_listener_recording(recording, (10.5, 3.0, 0.0))
    plain = render_recording(base_rec, block_size=1024)

    # Panels in room1 that keep the low band but absorb the high band.
    band_scene = build_corridor_scene()
    band_scene.zones.append(
        AbsorptionZone("hf_panels", (6.5, 2.5, -2.0), (7.5, 4.5, 2.0),
                       bands=[0.9, 0.15])
    )
    treated = render_recording(base_rec, block_size=1024, scene=band_scene)

    # Same panels, but the high-band material is swapped open at t=5:
    # sound produced earlier keeps the dull treatment, later sound clears.
    swap_scene = build_corridor_scene()
    swap_scene.zones.append(
        AbsorptionZone("swap_panels", (6.5, 2.5, -2.0), (7.5, 4.5, 2.0),
                       bands=[0.9, [(0.0, 0.15), (5.0, 0.15), (6.0, 0.9)]])
    )
    swapped = render_recording(base_rec, block_size=1024, scene=swap_scene)

    sf.write(OUT / "band_room2_plain.wav", plain.samples, SR, subtype="PCM_16")
    sf.write(OUT / "band_room2_treated.wav", treated.samples, SR, subtype="PCM_16")
    sf.write(OUT / "band_room2_swap.wav", swapped.samples, SR, subtype="PCM_16")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, (title, pick) in zip(axes, [("low band (<1 kHz)", 0),
                                        ("high band (>1 kHz)", 1)]):
        for label, res, style in [("original", plain, "-"),
                                  ("band panels", treated, "--"),
                                  ("panels swapped at 5 s", swapped, ":")]:
            low, high = _band_split(res.samples)
            t, rms = channel_rms(low if pick == 0 else high)
            ax.plot(t, rms[:, 0], style, label=label)
        ax.set_ylabel("RMS level")
        ax.set_title(f"Heard from room2: {title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "band_check.png", dpi=130)

    info = band_scene.explain_path((10.5, 3.0, 0.0), (4.5, 3.0, 0.0), 3.0)
    print(f"wrote band_room2_*.wav and band_check.png")
    print(f"path trace at t=3 s: {info}")

    def level(samples, t0, t1):
        seg = samples[int(t0 * SR):int(t1 * SR)]
        return float(np.sqrt(np.mean(seg ** 2)))
    plain_low, plain_high = _band_split(plain.samples)
    treated_low, treated_high = _band_split(treated.samples)
    # Detour-heavy window (door2 shut): panels cross the path, so the
    # high band drops clearly while the low band is mostly kept.
    assert level(treated_high, 2.5, 3.5) < 0.7 * level(plain_high, 2.5, 3.5)
    assert level(treated_low, 2.5, 3.5) > 0.6 * level(plain_low, 2.5, 3.5)
    print("band absorption checks passed.")


PANEL_DURATION = 4.0


def _panel_scene():
    room = Room("hall", (-8.0, -4.0, -2.0), (4.0, 4.0, 2.0),
                reverb_time=1.2, reverb_send=0.6)
    panel = Panel(
        "moving_band_panel",
        width=1.4,
        height=3.0,
        position_keys=[
            (0.0, (-1.0, 0.0, 0.0)),
            (1.4, (-1.0, 0.0, 0.0)),
            (2.4, (-1.0, -4.0, 0.0)),
        ],
        heading_keys=[
            (0.0, 0.0),
            (1.4, 0.0),
            (2.4, 1.1),
        ],
        bands=[0.9, 0.25, 0.05],
        revisions=[PanelRevision(
            "high_band_reopened",
            start=1.4,
            end=2.4,
            bands=[None, None, [(1.4, 0.05), (1.8, 0.9)]],
        )],
    )
    return RoomScene([room], band_edges=(500.0, 2000.0), panels=[panel]), panel


def _panel_recording(scene):
    rec = MotionRecording(
        duration=PANEL_DURATION,
        sample_rate=SR,
        listener_trajectory=Trajectory.linear([(0.0, (-6.0, 0.0, 0.0))]),
        listener_forward=(1.0, 0.0, 0.0),
        scene=scene,
        emitters=[
            EmitterSpec(
                name="crossing_route",
                kind="mix_tones",
                params={"duration": PANEL_DURATION, "fade": 0.005,
                         "tones": [(200.0, 0.28), (1000.0, 0.28),
                                   (5000.0, 0.28)]},
                trajectory=Trajectory.linear([(0.0, (0.0, 0.0, 0.0))]),
                events=[(1.0, False)],
            ),
            EmitterSpec(
                name="other_route",
                kind="mix_tones",
                params={"duration": PANEL_DURATION, "fade": 0.005,
                         "tones": [(200.0, 0.28), (1000.0, 0.28),
                                   (5000.0, 0.28)]},
                trajectory=Trajectory.linear([(0.0, (0.0, 3.0, 0.0))]),
                events=[(3.0, False)],
            ),
        ],
    )
    return rec


def demo_moving_panel():
    """Moving/oriented 3-band panel, interval revision, two comparison takes."""
    scene, panel = _panel_scene()
    recording = _panel_recording(scene)
    recording.save(OUT / "panel_recording.json")

    # History-preserving render: the source on the crossing route stops at
    # 1 s; its tail keeps the old panel basis after the panel slides away.
    history = render_recording(recording, block_size=1024)
    sf.write(OUT / "panel_history.wav", history.samples, SR, subtype="PCM_16")

    # Boundary-recompute comparison: fast-forward to the revision boundary,
    # clear only reverb buses, then continue. Old tails are intentionally
    # absent, making the listening difference directly audible.
    engine = build_engine(recording)
    recomputed_tail = engine.render_recomputed_from(
        1.4, PANEL_DURATION - 1.4, block_size=1024, reset_reverbs=True
    ).samples
    recomputed = np.zeros_like(history.samples)
    recomputed[:int(1.4 * SR)] = history.samples[:int(1.4 * SR)]
    recomputed[int(1.4 * SR):] = recomputed_tail
    sf.write(OUT / "panel_recomputed_boundary.wav", recomputed, SR,
             subtype="PCM_16")

    # Isolate the two geometric routes for A/B listening.
    crossing_rec = MotionRecording(
        PANEL_DURATION, SR, recording.listener_trajectory,
        recording.listener_forward, [recording.emitters[0]], scene=scene,
    )
    other_rec = MotionRecording(
        PANEL_DURATION, SR, recording.listener_trajectory,
        recording.listener_forward, [recording.emitters[1]], scene=scene,
    )
    crossing = render_recording(crossing_rec, block_size=1024)
    other = render_recording(other_rec, block_size=1024)
    sf.write(OUT / "panel_route_crossing.wav", crossing.samples, SR,
             subtype="PCM_16")
    sf.write(OUT / "panel_route_other.wav", other.samples, SR,
             subtype="PCM_16")

    # Block-size invariance of the complete moving/revised scenario.
    reference = history.samples
    block_diffs = {}
    for block_size in (37, 256, 4099):
        out = render_recording(recording, block_size=block_size).samples
        block_diffs[block_size] = float(np.max(np.abs(out - reference)))

    # Pause/resume at the revision boundary and an interval render both
    # match the same samples of the uninterrupted history render.
    chunk_engine = build_engine(recording)
    before_pause = chunk_engine.render_chunk(1.4).samples
    save_checkpoint(recording, chunk_engine, OUT / "panel_checkpoint.json")
    after_pause = chunk_engine.render_chunk(PANEL_DURATION - 1.4).samples
    _, resumed_engine = load_checkpoint(OUT / "panel_checkpoint.json")
    resumed = resumed_engine.render_chunk(PANEL_DURATION - 1.4).samples
    pause_diff = float(np.max(np.abs(resumed - after_pause)))
    interval = build_engine(recording).render_interval(
        0.7, 2.6, block_size=512
    ).samples
    interval_diff = float(np.max(np.abs(
        interval - reference[int(0.7 * SR):int(2.6 * SR)]
    )))

    manifest = panel_version_manifest(recording)
    with open(OUT / "panel_versions.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    t, rms = channel_rms(history.samples)
    _, rms_cross = channel_rms(crossing.samples)
    _, rms_other = channel_rms(other.samples)
    _, rms_recompute = channel_rms(recomputed)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, rms[:, 0], label="history (old tail retained)", lw=1.5)
    ax.plot(t, rms_recompute[:, 0], "--",
            label="recomputed from boundary (old tail cleared)")
    ax.plot(t, rms_cross[:, 0], ":", label="crossing route only")
    ax.plot(t, rms_other[:, 0], ":", label="other route only")
    for x, label in [(1.0, "crossing source off"), (1.4, "revision starts"),
                     (2.4, "panel leaves / revision ends"), (3.0, "other off")]:
        ax.axvline(x, color="gray", ls=":", lw=0.8)
        ax.text(x, ax.get_ylim()[1] * 0.92, label, rotation=90, va="top",
                fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMS level")
    ax.set_title("Moving oriented 3-band panel: history vs boundary recompute")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "panel_check.png", dpi=130)

    assert max(block_diffs.values()) < 1e-10
    assert pause_diff < 1e-10 and interval_diff < 1e-10
    print(f"wrote panel_history.wav / panel_recomputed_boundary.wav and route A/B WAVs")
    print(f"panel version manifest: {OUT / 'panel_versions.json'}")
    print(f"block-size max differences: {block_diffs}")
    print(f"pause/resume diff: {pause_diff:.3e}, interval diff: {interval_diff:.3e}")


if __name__ == "__main__":
    main()
    demo_corridor()
    demo_bands()
    demo_moving_panel()
