"""Rooms, doors (portals) and per-room reverbs for occlusion-aware rendering.

Geometry model:

- Rooms are axis-aligned boxes. A point is "inside" at most one room
  (first match wins); points outside every room form one shared "outside".
- A Portal connects two rooms and has a DoorMotion: keyframed open
  fraction in [0, 1], smoothstep-interpolated so volume/spectrum changes
  are continuous in time and independent of the render block size.
- Direct-path transmission between the listener room and a source room:
  same room -> (gain 1, no muffling); through a portal -> gain and
  lowpass mix follow the open fraction; no connecting portal -> heavy
  occlusion.

Reverbs are built from feedback comb + allpass filters implemented with
scipy lfilter and carried zi state, so reverb tails are sample-exact
regardless of how the render is split into blocks.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


class DoorMotion:
    """Keyframed door opening: sorted (time, open_fraction) keys.

    Fractions are smoothstep-interpolated between keys and held constant
    outside the key range, so the opening is a continuous function of time.
    """

    def __init__(self, keys=((0.0, 1.0),)):
        keys = sorted((float(t), float(f)) for t, f in keys)
        if not keys:
            keys = [(0.0, 1.0)]
        self._times = np.array([k[0] for k in keys], dtype=np.float64)
        self._fracs = np.clip(np.array([k[1] for k in keys], dtype=np.float64), 0.0, 1.0)

    def open_fraction(self, t):
        """Open fraction in [0, 1] at scalar or array-like time t."""
        scalar = np.isscalar(t)
        tv = np.atleast_1d(np.asarray(t, dtype=np.float64))
        times, fracs = self._times, self._fracs
        if len(times) == 1:
            out = np.full(len(tv), fracs[0])
            return float(out[0]) if scalar else out
        idx = np.clip(np.searchsorted(times, tv, side="right") - 1, 0, len(times) - 2)
        t0, t1 = times[idx], times[idx + 1]
        f0, f1 = fracs[idx], fracs[idx + 1]
        u = np.clip((tv - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
        s = u * u * (3.0 - 2.0 * u)
        out = f0 + (f1 - f0) * s
        out[tv <= times[0]] = fracs[0]
        out[tv >= times[-1]] = fracs[-1]
        return float(out[0]) if scalar else out

    def to_list(self):
        return [[float(t), float(f)] for t, f in zip(self._times, self._fracs)]

    @classmethod
    def from_list(cls, rows):
        return cls([(r[0], r[1]) for r in rows])


class Room:
    """Axis-aligned room box with its own reverb characteristics."""

    def __init__(self, name, min_corner, max_corner, reverb_time=0.6, reverb_send=0.25):
        self.name = str(name)
        self.min = np.asarray(min_corner, dtype=np.float64)
        self.max = np.asarray(max_corner, dtype=np.float64)
        if self.min.shape != (3,) or self.max.shape != (3,) or np.any(self.min >= self.max):
            raise ValueError("room corners must be length-3 with min < max")
        self.reverb_time = float(reverb_time)
        self.reverb_send = float(reverb_send)

    def mask(self, positions):
        """Boolean mask of points (n, 3) lying inside this room."""
        p = np.asarray(positions, dtype=np.float64)
        return np.all((p >= self.min) & (p <= self.max), axis=1)

    def to_dict(self):
        return {
            "name": self.name,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "reverb_time": self.reverb_time,
            "reverb_send": self.reverb_send,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["min"], d["max"],
                   d.get("reverb_time", 0.6), d.get("reverb_send", 0.25))


class Portal:
    """A door connecting two rooms; openness varies with DoorMotion."""

    def __init__(self, name, room_a, room_b, motion=None):
        self.name = str(name)
        self.room_a = str(room_a)
        self.room_b = str(room_b)
        self.motion = motion if motion is not None else DoorMotion()

    def connects(self, a, b):
        return {a, b} == {self.room_a, self.room_b}

    def to_dict(self):
        return {
            "name": self.name,
            "rooms": [self.room_a, self.room_b],
            "motion": self.motion.to_list(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["rooms"][0], d["rooms"][1],
                   DoorMotion.from_list(d.get("motion", [[0.0, 1.0]])))


class RoomScene:
    """Rooms + portals; answers direct/reverb transmission queries."""

    # Direct path through a portal: closed door keeps only a quiet,
    # muffled leakage; open door is transparent.
    DIRECT_CLOSED_GAIN = 0.08
    DIRECT_NO_PATH_GAIN = 0.02
    # Reverb heard across rooms: a closed door still lets the other
    # room's tail through clearly enough to be distinguishable.
    REVERB_CLOSED_GAIN = 0.3
    REVERB_NO_PATH_GAIN = 0.05

    def __init__(self, rooms, portals=()):
        self.rooms = list(rooms)
        self.portals = list(portals)
        self._index = {r.name: i for i, r in enumerate(self.rooms)}

    def room_index_at(self, positions):
        """Room index per point (n, 3); -1 when outside every room."""
        p = np.asarray(positions, dtype=np.float64)
        idx = np.full(len(p), -1, dtype=np.int64)
        unset = np.ones(len(p), dtype=bool)
        for i, room in enumerate(self.rooms):
            m = unset & room.mask(p)
            idx[m] = i
            unset[m] = False
        return idx

    def direct_transmission(self, listener_idx, source_idx, times):
        """Per-sample (gain, muffle_mix) for the direct path.

        muffle_mix in [0, 1] crossfades to a fixed lowpassed copy of the
        signal, so the spectrum changes continuously with door motion.
        """
        n = len(times)
        gain = np.ones(n)
        mix = np.zeros(n)
        diff = listener_idx != source_idx
        if diff.any():
            gain[diff] = self.DIRECT_NO_PATH_GAIN
            mix[diff] = 1.0
            for portal in self.portals:
                ia = self._index.get(portal.room_a, -2)
                ib = self._index.get(portal.room_b, -2)
                m = diff & (
                    ((listener_idx == ia) & (source_idx == ib))
                    | ((listener_idx == ib) & (source_idx == ia))
                )
                if m.any():
                    op = portal.motion.open_fraction(times[m])
                    gain[m] = self.DIRECT_CLOSED_GAIN + (1.0 - self.DIRECT_CLOSED_GAIN) * op
                    mix[m] = 1.0 - op
        return gain, mix

    def reverb_transmission(self, listener_idx, room_i, times):
        """Per-sample gain at which room_i's reverb reaches the listener."""
        n = len(times)
        gain = np.full(n, self.REVERB_NO_PATH_GAIN)
        same = listener_idx == room_i
        gain[same] = 1.0
        rest = ~same
        if rest.any():
            for portal in self.portals:
                ia = self._index.get(portal.room_a, -2)
                ib = self._index.get(portal.room_b, -2)
                if room_i == ia:
                    m = rest & (listener_idx == ib)
                elif room_i == ib:
                    m = rest & (listener_idx == ia)
                else:
                    continue
                if m.any():
                    op = portal.motion.open_fraction(times[m])
                    gain[m] = self.REVERB_CLOSED_GAIN + (1.0 - self.REVERB_CLOSED_GAIN) * op
        return gain

    def to_dict(self):
        return {
            "rooms": [r.to_dict() for r in self.rooms],
            "portals": [p.to_dict() for p in self.portals],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            [Room.from_dict(r) for r in d.get("rooms", [])],
            [Portal.from_dict(p) for p in d.get("portals", [])],
        )


class RoomReverb:
    """Stereo feedback-comb reverb with carried filter state.

    All filters run through scipy lfilter with zi saved between blocks,
    so the tail is bit-identical for any block size.
    """

    _COMB_DELAYS = (0.0297, 0.0371, 0.0411, 0.0437)
    _ALLPASS_DELAYS = (0.0050, 0.0017)
    _ALLPASS_GAIN = 0.5

    def __init__(self, sample_rate, reverb_time=0.6, channel_scales=(1.0, 1.13)):
        self._channels = []
        for scale in channel_scales:
            combs = []
            for d in self._COMB_DELAYS:
                delay = max(1, int(round(d * scale * sample_rate)))
                g = 10.0 ** (-3.0 * delay / (sample_rate * max(reverb_time, 1e-3)))
                a = np.zeros(delay + 1)
                a[0] = 1.0
                a[delay] = -g
                combs.append(([1.0], a, np.zeros(delay)))
            allpasses = []
            for d in self._ALLPASS_DELAYS:
                delay = max(1, int(round(d * scale * sample_rate)))
                g = self._ALLPASS_GAIN
                b = np.zeros(delay + 1)
                b[0] = -g
                b[delay] = 1.0
                a = np.zeros(delay + 1)
                a[0] = 1.0
                a[delay] = -g
                allpasses.append((b, a, np.zeros(delay)))
            self._channels.append((combs, allpasses))

    def process(self, mono):
        """mono (n,) -> wet stereo (n, 2); state carries across calls."""
        mono = np.asarray(mono, dtype=np.float64)
        out = np.empty((len(mono), 2))
        for ch, (combs, allpasses) in enumerate(self._channels):
            y = np.zeros(len(mono))
            for i, (b, a, zi) in enumerate(combs):
                c, zf = lfilter(b, a, mono, zi=zi)
                combs[i] = (b, a, zf)
                y += c
            y *= 0.25
            for i, (b, a, zi) in enumerate(allpasses):
                y, zf = lfilter(b, a, y, zi=zi)
                allpasses[i] = (b, a, zf)
            out[:, ch] = y
        return out

