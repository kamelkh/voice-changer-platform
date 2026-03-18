"""
Unit tests for the audio processing pipeline.
"""
from __future__ import annotations

import unittest

import numpy as np


def _sine(frames: int = 1024, sr: int = 44100, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, frames / sr, frames, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestAudioPipeline(unittest.TestCase):
    """Tests for :class:`~src.engine.pipeline.AudioPipeline`."""

    def _make_pipeline(self):
        from src.engine.pipeline import AudioPipeline
        return AudioPipeline()

    def test_empty_pipeline_passthrough(self) -> None:
        pipeline = self._make_pipeline()
        pipeline.input_gain = 1.0  # disable default +14 dB boost for passthrough test
        audio = _sine()
        result = pipeline.process(audio, 44100)
        np.testing.assert_array_almost_equal(result, audio)

    def test_bypass_mode(self) -> None:
        from src.engine.effects import PitchShifter
        pipeline = self._make_pipeline()
        pipeline.add_effect(PitchShifter(semitones=6.0))
        pipeline.bypass = True
        audio = _sine()
        result = pipeline.process(audio, 44100)
        np.testing.assert_array_almost_equal(result, audio)

    def test_effect_is_applied(self) -> None:
        from src.engine.effects import VolumeControl
        pipeline = self._make_pipeline()
        pipeline.add_effect(VolumeControl(gain=2.0))
        audio = _sine()
        result = pipeline.process(audio, 44100)
        # Result should differ (gain applied and clipped)
        self.assertFalse(np.allclose(result, audio))

    def test_add_remove_effect(self) -> None:
        from src.engine.effects import PitchShifter
        pipeline = self._make_pipeline()
        effect = PitchShifter(semitones=3.0)
        pipeline.add_effect(effect)
        self.assertEqual(len(pipeline.get_effects()), 1)
        pipeline.remove_effect(effect)
        self.assertEqual(len(pipeline.get_effects()), 0)

    def test_clear_effects(self) -> None:
        from src.engine.effects import PitchShifter, VolumeControl
        pipeline = self._make_pipeline()
        pipeline.add_effect(PitchShifter())
        pipeline.add_effect(VolumeControl())
        pipeline.clear_effects()
        self.assertEqual(len(pipeline.get_effects()), 0)

    def test_get_effect_by_type(self) -> None:
        from src.engine.effects import PitchShifter, VolumeControl
        pipeline = self._make_pipeline()
        pitch = PitchShifter(semitones=5.0)
        pipeline.add_effect(pitch)
        pipeline.add_effect(VolumeControl())
        found = pipeline.get_effect_by_type(PitchShifter)
        self.assertIs(found, pitch)

    def test_get_effect_type_not_present(self) -> None:
        from src.engine.effects import ReverbEffect
        pipeline = self._make_pipeline()
        self.assertIsNone(pipeline.get_effect_by_type(ReverbEffect))

    def test_chained_effects_shape_preserved(self) -> None:
        from src.engine.effects import (
            FormantShifter,
            NoiseGate,
            PitchShifter,
            ReverbEffect,
            VolumeControl,
        )
        pipeline = self._make_pipeline()
        pipeline.add_effect(NoiseGate(threshold_db=-80.0))
        pipeline.add_effect(PitchShifter(semitones=-3.0))
        pipeline.add_effect(FormantShifter(semitones=2.0))
        pipeline.add_effect(ReverbEffect(wet_level=0.2))
        pipeline.add_effect(VolumeControl(gain=1.1))
        audio = _sine(frames=2048)
        result = pipeline.process(audio, 44100)
        self.assertEqual(result.shape, audio.shape)

    def test_load_from_profile(self) -> None:
        from src.profiles.profile import VoiceProfile
        pipeline = self._make_pipeline()
        profile = VoiceProfile(
            name="Test",
            pitch_shift=-4.0,
            formant_shift=-2.0,
            reverb_level=0.1,
            noise_gate_threshold=-40.0,
            gain=1.2,
        )
        pipeline.load_from_profile(profile)
        # NoiseGate + PitchShifter + FormantShifter + Reverb + Compressor + VolumeControl = 6
        self.assertEqual(len(pipeline.get_effects()), 6)

    def test_last_process_time_updated(self) -> None:
        pipeline = self._make_pipeline()
        pipeline.process(_sine(), 44100)
        self.assertGreater(pipeline.last_process_time_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
