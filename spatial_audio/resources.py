"""Sound resources: immutable sample data shared between emitters."""

from __future__ import annotations

import numpy as np


class SoundResource:
    """A mono sample buffer with a sample rate.

    Resources are immutable; swapping the resource of an emitter does not
    affect other emitters using the same resource.
    """

    def __init__(self, samples: np.ndarray, sample_rate: int):
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError("SoundResource only supports mono samples")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._samples = data
        self.sample_rate = int(sample_rate)

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def duration(self) -> float:
        return len(self._samples) / self.sample_rate

    @staticmethod
    def sine_tone(
        frequency: float,
        duration: float,
        sample_rate: int,
        phase: float = 0.0,
        fade: float = 0.01,
    ) -> "SoundResource":
        """Deterministic sine tone with short edge fades (no clicks)."""
        n = int(round(duration * sample_rate))
        t = np.arange(n) / sample_rate
        data = np.sin(2.0 * np.pi * frequency * t + phase)
        if fade > 0 and n:
            k = min(int(round(fade * sample_rate)), n // 2)
            if k > 0:
                ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, k))
                data[:k] *= ramp
                data[-k:] *= ramp[::-1]
        return SoundResource(data, sample_rate)
