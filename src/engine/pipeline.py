"""
Audio processing pipeline.

Chains multiple :class:`~src.engine.effects.IAudioEffect` instances and
optionally feeds the audio into the RVC AI engine.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.engine.effects import (
    IAudioEffect,
    Compressor,
    NoiseGate,
    PitchShifter,
    FormantShifter,
    ReverbEffect,
    VoiceDisguise,
    VolumeControl,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioPipeline:
    """
    Chain effects and optionally apply AI voice conversion.

    Usage::

        pipeline = AudioPipeline()
        pipeline.add_effect(PitchShifter(semitones=-4))
        pipeline.add_effect(ReverbEffect(wet_level=0.1))
        output = pipeline.process(chunk, sample_rate)
    """

    def __init__(self) -> None:
        self._effects: list[IAudioEffect] = []
        self._rvc_engine: Optional[object] = None  # RVCEngine instance
        self._bypass: bool = False
        self._last_process_time_ms: float = 0.0

        # Input gain boost — compensates for quiet microphones (e.g. Galaxy
        # Buds Bluetooth at 16 kHz typically deliver RMS ~0.002 while normal
        # speech needs ~0.05).  Set via settings["processing"]["input_gain"].
        self.input_gain: float = 5.0  # +14 dB default

    # ── Effect management ─────────────────────────────────────────────────────

    def add_effect(self, effect: IAudioEffect) -> None:
        """Append an effect to the end of the chain."""
        self._effects.append(effect)
        logger.debug("Added effect: %s", effect.name)

    def remove_effect(self, effect: IAudioEffect) -> None:
        """Remove a specific effect instance from the chain."""
        self._effects = [e for e in self._effects if e is not effect]

    def clear_effects(self) -> None:
        """Remove all effects."""
        self._effects.clear()

    def get_effects(self) -> list[IAudioEffect]:
        """Return a copy of the current effects chain."""
        return list(self._effects)

    def get_effect_by_type(self, effect_type: type) -> Optional[IAudioEffect]:
        """Return the first effect of the given type, or *None*."""
        for effect in self._effects:
            if isinstance(effect, effect_type):
                return effect
        return None

    # ── RVC integration ───────────────────────────────────────────────────────

    def set_rvc_engine(self, engine: Optional[object]) -> None:
        """
        Attach an :class:`~src.engine.rvc_engine.RVCEngine` instance.

        When set, the RVC conversion is applied *before* the effects chain.
        Pass *None* to disable AI conversion.
        """
        self._rvc_engine = engine
        logger.info("RVC engine %s.", "attached" if engine else "detached")

    # ── Bypass ────────────────────────────────────────────────────────────────

    @property
    def bypass(self) -> bool:
        """When *True*, audio passes through unchanged."""
        return self._bypass

    @bypass.setter
    def bypass(self, value: bool) -> None:
        self._bypass = value
        logger.info("Pipeline bypass: %s", value)

    # ── Processing ────────────────────────────────────────────────────────────

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process a single audio chunk through the pipeline.

        Args:
            audio_data: Float32 NumPy array.
            sample_rate: Sample rate in Hz.

        Returns:
            Processed float32 NumPy array of the same shape.
        """
        t0 = time.perf_counter()

        if self._bypass:
            return audio_data

        result = audio_data.astype(np.float32)

        # ── Input gain boost ───────────────────────────────────────────
        # Galaxy Buds / Bluetooth mics are extremely quiet.  We amplify
        # early so every downstream effect gets a usable signal level.
        if self.input_gain != 1.0:
            result = np.clip(result * self.input_gain, -1.0, 1.0).astype(np.float32)

        # Measure input RMS *after* gain boost (this is the effective input)
        in_rms = float(np.sqrt(np.mean(result ** 2))) + 1e-10

        # AI voice conversion first
        if self._rvc_engine is not None:
            try:
                result = self._rvc_engine.convert(result, sample_rate)
            except Exception as exc:
                logger.error("RVC conversion error: %s", exc)

        # Then effects chain
        for effect in self._effects:
            try:
                result = effect.process(result, sample_rate)
            except Exception as exc:
                logger.error("Effect %s error: %s", effect.name, exc)

        # Pipeline-level RMS normalization: prevent cumulative signal loss
        # from multiple effects.  Only scale up (never attenuate) and only
        # when there is actual speech – skip for quiet/gated chunks to
        # avoid boosting noise.
        out_rms = float(np.sqrt(np.mean(result ** 2))) + 1e-10
        if in_rms > 0.005 and out_rms < in_rms * 0.5:
            # Signal lost more than 6 dB – restore to ~80 % of input RMS
            scale = (in_rms * 0.8) / out_rms
            result = np.clip(result * scale, -1.0, 1.0).astype(np.float32)

        self._last_process_time_ms = (time.perf_counter() - t0) * 1000.0
        return result

    @property
    def last_process_time_ms(self) -> float:
        """Processing time for the most recent chunk in milliseconds."""
        return self._last_process_time_ms

    # ── Profile loading ───────────────────────────────────────────────────────

    def load_from_profile(self, profile: "VoiceProfile") -> None:  # noqa: F821
        """
        Rebuild the effects chain from a :class:`~src.profiles.profile.VoiceProfile`.

        Args:
            profile: The voice profile to apply.
        """
        self.clear_effects()

        if profile.noise_gate_threshold > -80.0:
            self.add_effect(NoiseGate(threshold_db=profile.noise_gate_threshold))

        if profile.pitch_shift != 0.0:
            self.add_effect(PitchShifter(semitones=profile.pitch_shift))

        if profile.formant_shift != 0.0:
            self.add_effect(FormantShifter(semitones=profile.formant_shift))

        if getattr(profile, 'voice_disguise', 0.0) > 0.0:
            self.add_effect(VoiceDisguise(intensity=profile.voice_disguise))

        if profile.reverb_level > 0.0:
            self.add_effect(ReverbEffect(wet_level=profile.reverb_level))

        # Compressor evens out volume
        self.add_effect(Compressor(
            threshold_db=-25.0,
            ratio=3.0,
            attack_ms=5.0,
            release_ms=100.0,
            makeup_gain_db=4.0,   # +4 dB makeup
        ))

        if profile.gain != 1.0:
            self.add_effect(VolumeControl(gain=profile.gain))

        logger.info(
            "Pipeline loaded from profile '%s' (%d effects).",
            profile.name,
            len(self._effects),
        )
