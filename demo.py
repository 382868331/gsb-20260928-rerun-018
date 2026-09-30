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
    DoorMotion,
    MotionRecording,
    Portal,
    Room,
    RoomScene,
    Trajectory,
    render_recording,
)
from spatial_audio.recording import EmitterSpec

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


if __name__ == "__main__":
    main()
