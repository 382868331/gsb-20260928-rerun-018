from __future__ import annotations

import numpy as np


class SignalSource:
    """A reusable sound resource. Resources are independent of the sources
    that emit them and keep no engine-time state themselves."""

    def render(self, t0: float, n: int, sample_rate: int) -> np.ndarray:
        raise NotImplementedError


class SineTone(SignalSource):
    """Phase-continuous artificial tone, optionally with a slow vibrato."""

    def __init__(self, frequency: float, amplitude: float = 1.0, vibrato_hz: float = 0.0,
                 vibrato_depth: float = 0.0, phase: float = 0.0):
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.vibrato_hz = float(vibrato_hz)
        self.vibrato_depth = float(vibrato_depth)
        self.phase = float(phase)

    def render(self, t0: float, n: int, sample_rate: int) -> np.ndarray:
        t = t0 + np.arange(n, dtype=np.float64) / sample_rate
        if self.vibrato_hz > 0.0 and self.vibrato_depth > 0.0:
            inst_freq = self.frequency * (
                1.0 + self.vibrato_depth * np.sin(2.0 * np.pi * self.vibrato_hz * t)
            )
            phase = 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate
            phase += 2.0 * np.pi * self.frequency * t0 + self.phase
        else:
            phase = 2.0 * np.pi * self.frequency * t + self.phase
        return self.amplitude * np.sin(phase)
