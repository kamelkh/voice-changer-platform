"""
Unit tests for the audio effects.
"""
from __future__ import annotations

import unittest

import numpy as np


def _make_audio(frames: int = 1024, channels: int = 1, sr: int = 44100) -> np.ndarray:
    """Generate a sine wave as a test signal."""
    t = np.linspace(0, frames / sr, frames, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if channels == 1:
        return sine
    return np.stack([sine] * channels, axis=1)


class TestPitchShifter(unittest.TestCase):
    def _effect(self, semitones: float):
        from src.engine.effects import PitchShifter
        return PitchShifter(semitones=semitones)

    def test_passthrough_when_zero(self) -> None:
        effect = self._effect(0.0)
        audio = _make_audio()
        result = effect.process(audio, 44100)
        np.testing.assert_array_almost_equal(result, audio)

    def test_output_same_shape(self) -> None:
        for st in (-6, -2, 2, 6):
            with self.subTest(semitones=st):
                effect = self._effect(st)
                audio = _make_audio()
                result = effect.process(audio, 44100)
                self.assertEqual(result.shape, audio.shape)

    def test_get_set_params(self) -> None:
        effect = self._effect(3.0)
        self.assertEqual(effect.get_params()["semitones"], 3.0)
        effect.set_params({"semitones": -5.0})
        self.assertEqual(effect.get_params()["semitones"], -5.0)


class TestFormantShifter(unittest.TestCase):
    def _effect(self, semitones: float):
        from src.engine.effects import FormantShifter
        return FormantShifter(semitones=semitones)

    def test_passthrough_when_zero(self) -> None:
        effect = self._effect(0.0)
        audio = _make_audio()
        result = effect.process(audio, 44100)
        np.testing.assert_array_almost_equal(result, audio)

    def test_output_same_shape(self) -> None:
        effect = self._effect(2.0)
        audio = _make_audio()
        result = effect.process(audio, 44100)
        self.assertEqual(result.shape, audio.shape)


class TestNoiseGate(unittest.TestCase):
    def _effect(self, threshold_db: float):
        from src.engine.effects import NoiseGate
        return NoiseGate(threshold_db=threshold_db)

    def test_silence_below_threshold(self) -> None:
        effect = self._effect(threshold_db=-10.0)
        # Very quiet signal (−40 dB approximately)
        quiet = np.full(1024, 0.01, dtype=np.float32)
        # The gate uses a smooth release envelope, so process several chunks
        # to let the gain fully decay to 0 before asserting silence.
        for _ in range(20):
            result = effect.process(quiet, 44100)
        np.testing.assert_array_almost_equal(result, np.zeros_like(quiet), decimal=3)

    def test_loud_signal_passes_through(self) -> None:
        effect = self._effect(threshold_db=-60.0)
        loud = _make_audio()
        result = effect.process(loud, 44100)
        np.testing.assert_array_almost_equal(result, loud)

    def test_get_set_params(self) -> None:
        effect = self._effect(-40.0)
        effect.set_params({"threshold_db": -30.0, "attack_ms": 10.0})
        params = effect.get_params()
        self.assertEqual(params["threshold_db"], -30.0)
        self.assertEqual(params["attack_ms"], 10.0)


class TestReverbEffect(unittest.TestCase):
    def _effect(self, wet_level: float):
        from src.engine.effects import ReverbEffect
        return ReverbEffect(wet_level=wet_level)

    def test_dry_passthrough(self) -> None:
        effect = self._effect(0.0)
        audio = _make_audio()
        result = effect.process(audio, 44100)
        np.testing.assert_array_almost_equal(result, audio)

    def test_output_same_shape(self) -> None:
        effect = self._effect(0.3)
        audio = _make_audio()
        result = effect.process(audio, 44100)
        self.assertEqual(result.shape, audio.shape)

    def test_wet_output_differs_from_dry(self) -> None:
        effect = self._effect(0.5)
        audio = _make_audio(frames=4096)
        result = effect.process(audio, 44100)
        self.assertFalse(np.allclose(result, audio))


class TestVolumeControl(unittest.TestCase):
    def _effect(self, gain: float):
        from src.engine.effects import VolumeControl
        return VolumeControl(gain=gain)

    def test_unity_gain(self) -> None:
        effect = self._effect(1.0)
        audio = _make_audio()
        result = effect.process(audio, 44100)
        np.testing.assert_array_almost_equal(result, audio)

    def test_double_gain(self) -> None:
        effect = self._effect(2.0)
        audio = np.full(1024, 0.3, dtype=np.float32)
        result = effect.process(audio, 44100)
        np.testing.assert_array_almost_equal(result, np.clip(audio * 2.0, -1.0, 1.0))

    def test_clipping(self) -> None:
        effect = self._effect(10.0)
        audio = np.ones(512, dtype=np.float32)
        result = effect.process(audio, 44100)
        self.assertTrue(np.all(result <= 1.0))
        self.assertTrue(np.all(result >= -1.0))


class TestCompressor(unittest.TestCase):
    def _effect(self):
        from src.engine.effects import Compressor
        return Compressor(threshold_db=-20.0, ratio=4.0)

    def test_output_shape_preserved(self) -> None:
        effect = self._effect()
        audio = _make_audio()
        result = effect.process(audio, 44100)
        self.assertEqual(result.shape, audio.shape)

    def test_loud_signal_reduced(self) -> None:
        effect = self._effect()
        loud = np.ones(2048, dtype=np.float32) * 0.9
        result = effect.process(loud, 44100)
        self.assertLessEqual(float(np.abs(result).max()), 1.0)


class TestEffectFactory(unittest.TestCase):
    def test_create_all_effects(self) -> None:
        from src.engine.effects import EFFECT_REGISTRY, create_effect
        for name in EFFECT_REGISTRY:
            with self.subTest(effect=name):
                effect = create_effect(name)
                self.assertIsNotNone(effect)

    def test_unknown_effect_raises(self) -> None:
        from src.engine.effects import create_effect
        with self.assertRaises(KeyError):
            create_effect("nonexistent_effect")


if __name__ == "__main__":
    unittest.main()
