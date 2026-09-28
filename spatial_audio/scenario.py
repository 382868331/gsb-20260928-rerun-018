from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .listener import Listener
from .model import Vec3, Orientation
from .signals import SineTone
from .source import SoundSource
from .trajectory import LinearPath


@dataclass
class Scenario:
    duration: float
    sample_rate: int
    listener: Listener
    sources: list[SoundSource]

    def to_dict(self):
        f, _, u = self.listener.orientation.basis()
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "listener": {
                "position": [self.listener.position.x, self.listener.position.y, self.listener.position.z],
                "forward": f.tolist(),
                "up": u.tolist(),
            },
            "sources": [
                {
                    "name": s.name,
                    "gain": s.gain,
                    "active": s.active,
                    "deactivate_at": s.deactivate_at,
                    "tone": {
                        "frequency": s.signal.frequency,
                        "amplitude": s.signal.amplitude,
                        "vibrato_hz": s.signal.vibrato_hz,
                        "vibrato_depth": s.signal.vibrato_depth,
                    } if isinstance(s.signal, SineTone) else None,
                    "keyframes": s.trajectory.to_keyframes() if s.trajectory is not None else None,
                    "position": [s.position.x, s.position.y, s.position.z],
                }
                for s in self.sources
            ],
        }

    def save(self, path: str) -> str:
        payload = self.to_dict()
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        payload["sha256"] = digest
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return digest

    @classmethod
    def load(cls, path: str, verify: bool = True) -> "Scenario":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if verify and "sha256" in payload:
            digest = payload.pop("sha256")
            blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            actual = hashlib.sha256(blob).hexdigest()
            if actual != digest:
                raise ValueError(f"scenario checksum mismatch: {actual} != {digest}")
            payload["sha256"] = digest

        ld = payload["listener"]
        listener = Listener(Vec3(*ld["position"]), Orientation(Vec3(*ld["forward"]), Vec3(*ld["up"])))
        sources = []
        for sd in payload["sources"]:
            tone = None if sd["tone"] is None else SineTone(**sd["tone"])
            traj = None if sd["keyframes"] is None else LinearPath([tuple(k) for k in sd["keyframes"]])
            sources.append(
                SoundSource(
                    signal=tone,
                    position=Vec3(*sd["position"]),
                    trajectory=traj,
                    gain=sd.get("gain", 1.0),
                    active=sd.get("active", True),
                    name=sd.get("name", "source"),
                    deactivate_at=sd.get("deactivate_at"),
                )
            )
        return cls(payload["duration"], payload["sample_rate"], listener, sources)
