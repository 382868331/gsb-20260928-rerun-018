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
    load_session,
    render_interval,
    render_recording,
    save_session,
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


CORRIDOR_DURATION = 10.0


def build_corridor_scene(treated=False, door_b_keys=None, door_c_keys=None):
    """Corridor -> room1 -> room2, with room3 as a detour loop.

    door_b (room1-room2) gradually closes while the source is still on
    its way; the detour corridor->room1->room3->room2 keeps carrying
    sound. With treated=True an absorption zone is added inside room1
    where the corridor<->room2 propagation passes.
    """
    if door_b_keys is None:
        door_b_keys = [(0.0, 1.0), (2.5, 1.0), (3.5, 0.0), (6.0, 0.0), (7.0, 1.0)]
    if door_c_keys is None:
        door_c_keys = [(0.0, 1.0)]
    scene = RoomScene(
        rooms=[
            Room("corridor", (-12.0, -1.5, -2.0), (0.0, 1.5, 2.0), reverb_time=0.5),
            Room("room1", (0.0, -4.0, -2.0), (6.0, 4.0, 2.0), reverb_time=0.8),
            Room("room2", (6.0, -4.0, -2.0), (12.0, 4.0, 2.0), reverb_time=1.0),
            Room("room3", (0.0, -9.0, -2.0), (12.0, -5.0, 2.0), reverb_time=0.6),
        ],
        portals=[
            Portal("door_a", "corridor", "room1", DoorMotion([(0.0, 1.0)])),
            Portal("door_b", "room1", "room2", DoorMotion(door_b_keys)),
            Portal("door_c", "room1", "room3", DoorMotion(door_c_keys)),
            Portal("door_d", "room3", "room2", DoorMotion([(0.0, 1.0)])),
        ],
        zones=(
            [AbsorptionZone("treatment_room1", (1.5, -1.0, -2.0), (4.5, 1.0, 2.0),
                            absorption=0.75)]
            if treated else []
        ),
    )
    return scene


def build_corridor_recording(treated=False, listener="room2", **scene_kwargs):
    """One walking take: source corridor -> room1 -> room2; the listener
    waits in room2 (or sits still in the corridor for a second take)."""
    scene = build_corridor_scene(treated=treated, **scene_kwargs)
    if listener == "room2":
        listener_trajectory = Trajectory.linear(
            [(0.0, (10.0, 2.0, 0.0)), (CORRIDOR_DURATION, (9.0, 1.5, 0.0))]
        )
        listener_forward = (-1.0, 0.0, 0.0)
    else:
        listener_trajectory = Trajectory.linear([(0.0, (-6.0, 0.0, 0.0))])
        listener_forward = (1.0, 0.0, 0.0)
    return MotionRecording(
        duration=CORRIDOR_DURATION,
        sample_rate=SR,
        listener_trajectory=listener_trajectory,
        listener_forward=listener_forward,
        scene=scene,
        emitters=[
            EmitterSpec(
                name="walker",
                kind="tone",
                params={"frequency": 500.0, "duration": CORRIDOR_DURATION,
                        "fade": 0.01},
                trajectory=Trajectory.linear(
                    [
                        (0.0, (-10.0, 0.0, 0.0)),    # far down the corridor
                        (2.5, (-1.0, 0.0, 0.0)),     # reaches door_a
                        (4.0, (3.0, 0.0, 0.0)),      # inside room1
                        (6.0, (5.5, -0.5, 0.0)),     # waits at closed door_b
                        (8.0, (9.0, 0.0, 0.0)),      # enters room2
                        (CORRIDOR_DURATION, (9.0, 0.0, 0.0)),
                    ]
                ),
                events=[(8.0, False)],   # source stops; tails keep decaying
            ),
        ],
    )


def _rms_db(samples, hop_ms=50.0):
    hop = int(SR * hop_ms / 1000.0)
    n = len(samples) // hop
    frames = samples[: n * hop].reshape(n, hop, 2)
    rms = np.sqrt(np.mean(frames ** 2, axis=(1, 2)))
    t = (np.arange(n) + 0.5) * hop_ms / 1000.0
    return t, 20.0 * np.log10(np.maximum(rms, 1e-7))


def demo_corridor():
    # 1) One recorded walk, rendered with the base door arrangement.
    rec = build_corridor_recording()
    rec.save(OUT / "corridor_recording.json")
    full = render_recording(rec, block_size=1024)
    sf.write(OUT / "demo_corridor.wav", full.samples, SR, subtype="PCM_16")

    # 2) Same recording, replayed with a different door arrangement:
    # door_b stays open but the detour door_c is closed instead.
    alt = build_corridor_recording(
        door_b_keys=[(0.0, 1.0)],
        door_c_keys=[(0.0, 0.0)],
    )
    alt_render = render_recording(alt, block_size=777)
    sf.write(OUT / "corridor_replay_alt_doors.wav", alt_render.samples, SR,
             subtype="PCM_16")
    door_diff = float(np.max(np.abs(full.samples - alt_render.samples)))
    assert door_diff > 1e-3

    # 3) Save mid-render and resume later: bit-identical to the full take.
    engine = build_engine(rec)
    first = engine.render(5.0, block_size=1024)
    save_session(OUT / "corridor_session.json", rec, engine, 5.0)
    rec2, engine2, t_resume = load_session(OUT / "corridor_session.json")
    assert abs(t_resume - 5.0) < 1e-12
    second = engine2.render(rec2.duration - t_resume, block_size=1024,
                            start_time=t_resume)
    resumed = np.concatenate([first.samples, second.samples], axis=0)
    sf.write(OUT / "corridor_resumed.wav", resumed, SR, subtype="PCM_16")
    resume_diff = float(np.max(np.abs(resumed - full.samples)))
    assert resume_diff < 1e-9

    # 4) A selected interval matches the same slice of the full render.
    window = render_interval(rec, 2.0, 4.0, block_size=333)
    ref_slice = full.samples[int(2.0 * SR):int(4.0 * SR)]
    interval_diff = float(np.max(np.abs(window.samples - ref_slice)))
    assert interval_diff < 1e-9

    # 5) Absorption retrofit: same walking take, original walls vs added
    # absorption zone, compared at two listening positions.
    curves = {}
    for label, treated, listener in [
        ("room2 listener, original walls", False, "room2"),
        ("room2 listener, absorbed zone", True, "room2"),
        ("corridor listener, original walls", False, "corridor"),
        ("corridor listener, absorbed zone", True, "corridor"),
    ]:
        r = render_recording(
            build_corridor_recording(treated=treated, listener=listener),
            block_size=1024,
        )
        curves[label] = r.samples
    sf.write(OUT / "corridor_treated.wav",
             curves["room2 listener, absorbed zone"], SR, subtype="PCM_16")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {
        "room2 listener, original walls": "#1f77b4",
        "room2 listener, absorbed zone": "#d62728",
        "corridor listener, original walls": "#2ca02c",
        "corridor listener, absorbed zone": "#9467bd",
    }
    for label, samples in curves.items():
        t, db = _rms_db(samples)
        axes[0].plot(t, db, label=label, color=colors[label], lw=1.0)
    for x, text_label in [(3.5, "door_b shut"), (7.0, "door_b open"),
                          (8.0, "source off")]:
        axes[0].axvline(x, color="gray", ls="--", lw=0.8)
        axes[0].text(x, -5, text_label, rotation=90, va="top", fontsize=8)
    axes[0].set_ylabel("level (dB RMS)")
    axes[0].set_title("Absorption retrofit at two listening positions")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Decay after the source stops (t=8): the change is path-specific -
    # at the corridor listener the treated tail drops (its path from the
    # far rooms crosses the zone), at the room2 listener it is unchanged.
    for label in curves:
        t, db = _rms_db(curves[label])
        m = t >= 8.0
        axes[1].plot(t[m], db[m], label=label, color=colors[label], lw=1.2)
    axes[1].set_ylabel("decay after stop (dB)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "absorption_compare.png", dpi=130)

    def level(samples, t0, t1):
        seg = samples[int(t0 * SR):int(t1 * SR)]
        return float(np.sqrt(np.mean(seg ** 2)))

    base = curves["room2 listener, original walls"]
    treated = curves["room2 listener, absorbed zone"]
    # Source in the corridor (path crosses the zone): clearly quieter.
    assert level(treated, 1.0, 2.4) < 0.75 * level(base, 1.0, 2.4)
    # Source next to the listener in room2 (path misses the zone):
    # nearly unchanged - absorption is path-specific, not global.
    near_ratio = level(treated, 7.2, 7.9) / max(level(base, 7.2, 7.9), 1e-12)
    assert 0.9 < near_ratio < 1.01
    # Corridor listener, source still in the corridor (path misses the
    # zone): unchanged. Once the source is inside room1 the corridor
    # listener's path crosses the zone and drops - the zone only affects
    # propagation that passes through it.
    c_base = curves["corridor listener, original walls"]
    c_treated = curves["corridor listener, absorbed zone"]
    c_ratio = level(c_treated, 1.0, 2.4) / max(level(c_base, 1.0, 2.4), 1e-12)
    assert c_ratio > 0.95
    c_cross = level(c_treated, 4.2, 5.8) / max(level(c_base, 4.2, 5.8), 1e-12)
    assert c_cross < 0.5
    # Tails still decaying after the source stops in both schemes.
    assert level(base, 8.3, 8.8) > 1e-4
    assert level(treated, 8.3, 8.8) > 1e-5

    print(f"wrote {OUT / 'demo_corridor.wav'} (corridor -> room1 -> room2)")
    print(f"alt door arrangement max |difference|: {door_diff:.3e}")
    print(f"mid-render save/resume max |difference|: {resume_diff:.3e}")
    print(f"interval vs full-render slice max |difference|: {interval_diff:.3e}")
    print("corridor multi-path, interval and absorption checks passed.")


if __name__ == "__main__":
    main()
    demo_corridor()
