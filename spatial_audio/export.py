from __future__ import annotations

import numpy as np
import soundfile as sf


def export_wav(path: str, audio: np.ndarray, sample_rate: int, normalize: bool = True,
               target_peak: float = 0.9) -> dict:
    """Write a stereo float array to a 16-bit WAV.

    Returns metadata including whether normalization was needed. Mixing is
    already soft-limited; normalization only sets a safe final level.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    scale = 1.0
    if normalize and peak > 0:
        scale = min(1.0, target_peak / peak)
    pcm = np.round(audio * scale * 32767.0).astype(np.int16)
    sf.write(path, pcm, sample_rate, subtype="PCM_16")
    return {
        "path": path,
        "samples": int(audio.shape[0]),
        "duration_s": audio.shape[0] / sample_rate,
        "peak_float": peak,
        "scale": scale,
    }
