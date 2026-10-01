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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from spatial_audio import (
    AbsorptionZone,
    DoorMotion,
    MotionRecording,
    Portal,
    Room,
    RoomScene,
    Trajectory,
    load_checkpoint,
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


if __name__ == "__main__":
    main()
    demo_corridor()
