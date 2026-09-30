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
    Door,
    MotionRecording,
    Occluder,
    Room,
    Trajectory,
    render_recording,
)
from spatial_audio.recording import EmitterSpec

SR = 48000
DURATION = 6.0
OUT = Path("outputs")


def build_recording():
    # Two rooms share a wall with a doorway around y in [-1.2, 1.2].
    room_a = Room("hall_A", (-12.0, -8.0), (0.0, 8.0), rt60=0.45)
    room_b = Room("hall_B", (0.0, -8.0), (12.0, 8.0), rt60=0.8)
    occluders = [
        Occluder("wall_low", (0.0, -8.0), (0.0, -1.2)),
        Occluder("wall_high", (0.0, 1.2), (0.0, 8.0)),
    ]
    door = Door("door", (0.0, -1.2), (0.0, 1.2), openness=1.0)
    door.schedule_openness(2.0, 0.0)
    door.schedule_openness(4.0, 1.0)
    occluders.append(door)

    # Main source: sweeps from room A, through the open doorway (y = 0) at
    # t = 3 s, into room B -- level and spectrum stay continuous.
    moving = Trajectory.linear(
        [
            (0.0, (-12.0, 0.0, 0.0)),
            (3.0, (0.0, 0.0, 0.0)),
            (6.0, (12.0, 0.0, 0.0)),
        ]
    )
    # Second source: stationary in room B. The door closes at t = 2 s and
    # reopens at t = 4 s (direct path blocked, room tail survives), then the
    # source itself switches off at t = 5 s. Positioned well off the sweep
    # line so it does not mask the final left-to-right pan check.
    right_beep = Trajectory.linear([(0.0, (8.0, -6.0, 0.0))])

    return MotionRecording(
        duration=DURATION,
        sample_rate=SR,
        listener_trajectory=None,
        listener_forward=(0.0, 1.0, 0.0),
        listener_position=(0.0, -3.0, 0.0),
        rooms=[room_a, room_b],
        occluders=occluders,
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
                events=[(4.5, False)],
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

    # Sanity: early on, the moving source is on the listener's left; the
    # door-close/reopen events are present in the saved recording.
    early = slice(int(2.4 * SR), int(2.8 * SR))   # moving source still left
    assert np.abs(result.samples[early, 0]).mean() > np.abs(result.samples[early, 1]).mean()
    assert any(
        type(w).__name__ == "Door" and w.events for w in recording.occluders
    )
    assert max_diff < 1e-9
    print("Leftward pan, doorway events and replay-equivalence checks passed.")


if __name__ == "__main__":
    main()
