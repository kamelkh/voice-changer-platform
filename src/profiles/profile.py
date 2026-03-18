"""
Voice profile data model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class VoiceProfile:
    """
    Holds all parameters for a named voice preset.

    Basic effects are applied by :class:`~src.engine.pipeline.AudioPipeline`.
    When ``use_ai`` is *True*, an RVC model is loaded before the effects chain.
    """

    # ── Identity ───────────────────────────────────────────────────────────────
    name: str = "Unnamed"
    description: str = ""
    icon: str = "🎙️"

    # ── Basic effects ──────────────────────────────────────────────────────────
    pitch_shift: float = 0.0             # semitones, −12 … +12
    formant_shift: float = 0.0          # semitones, −6 … +6
    reverb_level: float = 0.0           # 0.0 (dry) – 1.0 (full wet)
    noise_gate_threshold: float = -40.0  # dBFS
    gain: float = 1.0                    # linear multiplier

    # ── Voice disguise (identity masking) ────────────────────────────────────────
    voice_disguise: float = 0.0          # 0.0 (off) – 1.0 (maximum disguise)

    # ── Dialect accent ────────────────────────────────────────────────────────
    accent_dialect: str = "none"         # "none" | "palestinian" | "syrian" | "lebanese" | "egyptian"
    accent_intensity: float = 0.0        # 0.0 (off) – 1.0 (full accent)

    # ── AI voice conversion ────────────────────────────────────────────────────
    use_ai: bool = False
    ai_model_path: str = ""
    ai_index_rate: float = 0.5
    ai_pitch_shift: int = 0
    ai_protect: float = 0.33
    ai_f0_method: str = "rmvpe"
    ai_filter_radius: int = 3

    # ── Metadata ───────────────────────────────────────────────────────────────
    tags: list[str] = field(default_factory=list)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        """Write the profile to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    # ── Deserialisation ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        """
        Create a :class:`VoiceProfile` from a dict (e.g. loaded from JSON).

        Unknown keys are silently ignored so older profile files are
        still compatible with newer versions of the data class.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "VoiceProfile":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str | Path) -> "VoiceProfile":
        """Load a profile from a JSON file."""
        path = Path(path)
        return cls.from_json(path.read_text(encoding="utf-8"))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def copy(self) -> "VoiceProfile":
        """Return a shallow copy of this profile."""
        return VoiceProfile.from_dict(self.to_dict())

    def apply_overrides(self, **kwargs) -> "VoiceProfile":
        """
        Return a new profile with specific fields overridden.

        Example::

            louder = profile.apply_overrides(gain=2.0, reverb_level=0.3)
        """
        data = self.to_dict()
        data.update(kwargs)
        return VoiceProfile.from_dict(data)

    def __str__(self) -> str:  # noqa: D105
        ai_indicator = " [AI]" if self.use_ai else ""
        return f"{self.icon} {self.name}{ai_indicator}"
