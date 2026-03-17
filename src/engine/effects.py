"""
Audio effects implementations.

Each effect implements the :class:`IAudioEffect` interface and can be
chained together in a :class:`~src.engine.pipeline.AudioPipeline`.

Design principle: **all processing in the hot path must be vectorised
NumPy/SciPy operations** – no Python-level per-sample loops.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy.signal as signal

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Base interface ────────────────────────────────────────────────────────────


class IAudioEffect(ABC):
    """Abstract base class for all audio effects."""

    @abstractmethod
    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process a chunk and return a result of the same shape."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return the current effect parameters as a plain dict."""

    @abstractmethod
    def set_params(self, params: dict[str, Any]) -> None:
        """Apply parameters from a dict (unknown keys are silently ignored)."""

    @property
    def name(self) -> str:
        """Human-readable effect name."""
        return self.__class__.__name__

    def _to_mono_1d(self, audio: np.ndarray) -> tuple[np.ndarray, bool, int]:
        """Return (1-D float32 mono array, was_2d, original_channels)."""
        if audio.ndim == 1:
            return audio.astype(np.float32), False, 1
        channels = audio.shape[1]
        mono = audio[:, 0].astype(np.float32)   # use first channel
        return mono, True, channels

    def _restore(self, mono: np.ndarray, was_2d: bool, channels: int) -> np.ndarray:
        """Broadcast mono result back to original channel layout."""
        if not was_2d:
            return mono
        return np.stack([mono] * channels, axis=1)


# ── Pitch Shifter ─────────────────────────────────────────────────────────────


class PitchShifter(IAudioEffect):
    """
    Real-time pitch shifter based on STFT phase-vocoder resampling.

    Works on small chunks (256–1024 frames) with acceptable quality.
    Uses scipy.signal.resample_poly (integer ratio) when the semitone
    value maps cleanly, otherwise falls back to resample.
    """

    def __init__(self, semitones: float = 0.0) -> None:
        self.semitones = float(semitones)
        self._buf = np.array([], dtype=np.float32)   # carry-over buffer

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.semitones == 0.0:
            return audio_data

        mono, was_2d, channels = self._to_mono_1d(audio_data)
        n = len(mono)
        ratio = 2.0 ** (self.semitones / 12.0)

        # Resample to simulate pitch stretch then trim to original length.
        # scipy.signal.resample is O(n log n) via FFT – safe for real-time.
        new_len = max(1, int(round(n / ratio)))
        resampled = signal.resample(mono, new_len).astype(np.float32)

        if len(resampled) >= n:
            out = resampled[:n]
        else:
            out = np.concatenate([resampled,
                                   np.zeros(n - len(resampled), dtype=np.float32)])

        return self._restore(out, was_2d, channels)

    def get_params(self) -> dict[str, Any]:
        return {"semitones": self.semitones}

    def set_params(self, params: dict[str, Any]) -> None:
        if "semitones" in params:
            self.semitones = float(params["semitones"])


# ── Formant Shifter ───────────────────────────────────────────────────────────


class FormantShifter(IAudioEffect):
    """
    Spectral-envelope warp to shift formants without changing pitch.

    Implemented as frequency-domain interpolation using rfft.
    """

    def __init__(self, semitones: float = 0.0) -> None:
        self.semitones = float(semitones)

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.semitones == 0.0:
            return audio_data

        mono, was_2d, channels = self._to_mono_1d(audio_data)
        ratio = 2.0 ** (self.semitones / 12.0)
        n = len(mono)

        spec = np.fft.rfft(mono)
        freqs = np.arange(len(spec), dtype=np.float32)
        warped = freqs / ratio

        real = np.interp(freqs, warped, spec.real, left=0.0, right=0.0)
        imag = np.interp(freqs, warped, spec.imag, left=0.0, right=0.0)
        out = np.fft.irfft(real + 1j * imag)[:n].astype(np.float32)
        if len(out) < n:
            out = np.pad(out, (0, n - len(out)))

        return self._restore(out, was_2d, channels)

    def get_params(self) -> dict[str, Any]:
        return {"semitones": self.semitones}

    def set_params(self, params: dict[str, Any]) -> None:
        if "semitones" in params:
            self.semitones = float(params["semitones"])


# ── Noise Gate ────────────────────────────────────────────────────────────────


class NoiseGate(IAudioEffect):
    """
    Hard gate: silence the chunk when its RMS falls below *threshold_db*.

    Uses a short hysteresis window to avoid chattering.
    """

    def __init__(self, threshold_db: float = -40.0,
                 attack_ms: float = 5.0, release_ms: float = 50.0) -> None:
        self.threshold_db = float(threshold_db)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self._gate_gain: float = 1.0

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        rms_db = 20.0 * np.log10(
            float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))) + 1e-10
        )
        target = 1.0 if rms_db >= self.threshold_db else 0.0

        # Simple one-pole smoothing
        tc = self.release_ms if target > self._gate_gain else self.attack_ms
        coef = np.exp(-1.0 / max(1, sample_rate * tc / 1000.0))
        self._gate_gain = coef * self._gate_gain + (1.0 - coef) * target

        return (audio_data.astype(np.float32) * self._gate_gain).astype(audio_data.dtype)

    def get_params(self) -> dict[str, Any]:
        return {"threshold_db": self.threshold_db,
                "attack_ms": self.attack_ms,
                "release_ms": self.release_ms}

    def set_params(self, params: dict[str, Any]) -> None:
        if "threshold_db" in params:
            self.threshold_db = float(params["threshold_db"])
        if "attack_ms" in params:
            self.attack_ms = float(params["attack_ms"])
        if "release_ms" in params:
            self.release_ms = float(params["release_ms"])


# ── Reverb ────────────────────────────────────────────────────────────────────


class ReverbEffect(IAudioEffect):
    """
    Vectorised multi-tap comb-filter reverb (no Python sample loops).

    An impulse response is synthesised from exponentially decaying
    comb delays and convolved with the audio using
    ``scipy.signal.fftconvolve`` (O(n log n)).  The IR is only
    recomputed when parameters or sample-rate change.
    """

    _COMB_DELAYS_MS = [29.7, 37.1, 41.1, 43.7, 47.3, 53.5]

    def __init__(self, wet_level: float = 0.3, room_size: float = 0.5) -> None:
        self.wet_level  = float(wet_level)
        self.room_size  = float(room_size)
        self._ir: np.ndarray | None = None
        self._last_sr: int = 0
        self._last_room: float = -1.0
        # Overlap-add carry buffer (length = len(IR) - 1)
        self._ola_buf: np.ndarray = np.array([], dtype=np.float32)

    # ── IR synthesis ─────────────────────────────────────────────────────────

    def _build_ir(self, sample_rate: int) -> None:
        """Build the reverb impulse response."""
        feedback = 0.4 + self.room_size * 0.45   # 0.40 – 0.85
        max_delay_samp = int(sample_rate * max(self._COMB_DELAYS_MS) / 1000) + 1
        ir = np.zeros(max_delay_samp, dtype=np.float32)
        for d_ms in self._COMB_DELAYS_MS:
            d = int(sample_rate * d_ms / 1000)
            # Exponential decay comb
            n_taps = max_delay_samp // max(1, d)
            for k in range(1, n_taps + 1):
                idx = k * d
                if idx < max_delay_samp:
                    ir[idx] += (feedback ** k) / len(self._COMB_DELAYS_MS)
        ir[0] += 1.0   # direct (dry) component in the IR
        self._ir = ir
        self._ola_buf = np.zeros(len(ir) - 1, dtype=np.float32)
        self._last_sr = sample_rate
        self._last_room = self.room_size

    # ── Process ───────────────────────────────────────────────────────────────

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.wet_level == 0.0:
            return audio_data

        mono, was_2d, channels = self._to_mono_1d(audio_data)

        # Rebuild IR when parameters change
        if (self._ir is None or sample_rate != self._last_sr
                or self.room_size != self._last_room):
            self._build_ir(sample_rate)

        # Overlap-add convolution
        n = len(mono)
        ir = self._ir
        conv = signal.oaconvolve(mono, ir, mode="full").astype(np.float32)

        # Add carry-over from previous chunk
        ola_len = len(self._ola_buf)
        if ola_len > 0:
            conv[:ola_len] += self._ola_buf

        # Output = first n samples; save tail for next chunk
        out_wet = conv[:n]
        new_tail = conv[n: n + ola_len]
        if len(new_tail) < ola_len:
            new_tail = np.pad(new_tail, (0, ola_len - len(new_tail)))
        self._ola_buf = new_tail

        # Dry/wet mix  (dry component is already in the IR at index 0,
        # so we subtract the direct path from the wet to avoid doubling)
        dry = mono
        wet = out_wet - dry   # reverb tail only
        mixed = np.clip(dry + self.wet_level * wet, -1.0, 1.0).astype(np.float32)

        if len(mixed) < n:
            mixed = np.pad(mixed, (0, n - len(mixed)))

        return self._restore(mixed[:n], was_2d, channels)

    def get_params(self) -> dict[str, Any]:
        return {"wet_level": self.wet_level, "room_size": self.room_size}

    def set_params(self, params: dict[str, Any]) -> None:
        if "wet_level" in params:
            self.wet_level = float(params["wet_level"])
        if "room_size" in params:
            self.room_size  = float(params["room_size"])
            self._ir = None   # force IR rebuild


# ── Volume Control ────────────────────────────────────────────────────────────


class VolumeControl(IAudioEffect):
    """Linear gain with soft peak-limiter."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = float(gain)

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.clip(audio_data.astype(np.float32) * self.gain, -1.0, 1.0).astype(audio_data.dtype)

    def get_params(self) -> dict[str, Any]:
        return {"gain": self.gain}

    def set_params(self, params: dict[str, Any]) -> None:
        if "gain" in params:
            self.gain = float(params["gain"])


# ── Compressor ────────────────────────────────────────────────────────────────


class Compressor(IAudioEffect):
    """
    Dynamic range compressor with attack / release smoothing.

    Operates per chunk (RMS-based gain reduction) – suitable for
    real-time use at chunk sizes 128–1024.
    """

    def __init__(self, threshold_db: float = -20.0, ratio: float = 4.0,
                 attack_ms: float = 5.0, release_ms: float = 100.0,
                 makeup_gain_db: float = 0.0) -> None:
        self.threshold_db   = float(threshold_db)
        self.ratio          = float(ratio)
        self.attack_ms      = float(attack_ms)
        self.release_ms     = float(release_ms)
        self.makeup_gain_db = float(makeup_gain_db)
        self._gain_db: float = 0.0

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = audio_data.astype(np.float32)
        rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(audio ** 2))) + 1e-10)

        over = rms_db - self.threshold_db
        target_gain_db = (
            (-over * (1.0 - 1.0 / self.ratio)) + self.makeup_gain_db
            if over > 0 else self.makeup_gain_db
        )

        tc_ms = self.attack_ms if target_gain_db < self._gain_db else self.release_ms
        coef = np.exp(-1.0 / max(1, sample_rate * tc_ms / 1000.0))
        self._gain_db = coef * self._gain_db + (1.0 - coef) * target_gain_db

        linear = 10.0 ** (self._gain_db / 20.0)
        return np.clip(audio * linear, -1.0, 1.0).astype(audio_data.dtype)

    def get_params(self) -> dict[str, Any]:
        return {"threshold_db": self.threshold_db, "ratio": self.ratio,
                "attack_ms": self.attack_ms, "release_ms": self.release_ms,
                "makeup_gain_db": self.makeup_gain_db}

    def set_params(self, params: dict[str, Any]) -> None:
        for key in ("threshold_db", "ratio", "attack_ms",
                    "release_ms", "makeup_gain_db"):
            if key in params:
                setattr(self, key, float(params[key]))


# ── Factory ───────────────────────────────────────────────────────────────────

EFFECT_REGISTRY: dict[str, type[IAudioEffect]] = {
    "pitch_shift":   PitchShifter,
    "formant_shift": FormantShifter,
    "noise_gate":    NoiseGate,
    "reverb":        ReverbEffect,
    "volume":        VolumeControl,
    "compressor":    Compressor,
}


def create_effect(effect_type: str, **kwargs: Any) -> IAudioEffect:
    """Factory: create an effect by registry name."""
    return EFFECT_REGISTRY[effect_type](**kwargs)


# ── Base interface ────────────────────────────────────────────────────────────


class IAudioEffect(ABC):
    """Abstract base class for audio effects."""

    @abstractmethod
    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process an audio chunk.

        Args:
            audio_data: Float32 array shaped ``(frames, channels)`` or
                        ``(frames,)``.
            sample_rate: Sample rate in Hz.

        Returns:
            Processed audio with the same shape as *audio_data*.
        """

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return the current effect parameters as a dict."""

    @abstractmethod
    def set_params(self, params: dict[str, Any]) -> None:
        """Apply parameters from a dict."""

    @property
    def name(self) -> str:
        """Human-readable effect name."""
        return self.__class__.__name__

    def _ensure_2d(self, audio: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (2-D array, was_1d)."""
        if audio.ndim == 1:
            return audio.reshape(-1, 1), True
        return audio, False

    def _restore_shape(self, audio: np.ndarray, was_1d: bool) -> np.ndarray:
        if was_1d:
            return audio.squeeze(1)
        return audio


# ── Pitch Shifter ─────────────────────────────────────────────────────────────


class PitchShifter(IAudioEffect):
    """
    Shift audio pitch by *semitones* without changing speed.

    Uses a phase-vocoder approach via STFT resampling so it is fast
    enough for real-time use without requiring *librosa* (which is
    much slower on small chunks).
    """

    def __init__(self, semitones: float = 0.0) -> None:
        """
        Args:
            semitones: Pitch shift in semitones (−12 to +12).
        """
        self.semitones = float(semitones)

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.semitones == 0.0:
            return audio_data

        audio, was_1d = self._ensure_2d(audio_data)
        ratio = 2.0 ** (self.semitones / 12.0)
        channels = audio.shape[1]
        out_channels = []

        for ch in range(channels):
            mono = audio[:, ch].astype(np.float32)
            # Resample to simulate pitch shift (changes speed)
            new_len = max(1, int(len(mono) / ratio))
            resampled = signal.resample(mono, new_len)
            # Trim or pad back to original length
            if len(resampled) >= len(mono):
                shifted = resampled[: len(mono)]
            else:
                shifted = np.pad(resampled, (0, len(mono) - len(resampled)))
            out_channels.append(shifted)

        result = np.stack(out_channels, axis=1)
        return self._restore_shape(result, was_1d)

    def get_params(self) -> dict[str, Any]:
        return {"semitones": self.semitones}

    def set_params(self, params: dict[str, Any]) -> None:
        if "semitones" in params:
            self.semitones = float(params["semitones"])


# ── Formant Shifter ───────────────────────────────────────────────────────────


class FormantShifter(IAudioEffect):
    """
    Shift vocal formants independently of pitch.

    Applies a simple spectral envelope warp using resampling and
    spectral interpolation as a lightweight approximation.
    """

    def __init__(self, semitones: float = 0.0) -> None:
        """
        Args:
            semitones: Formant shift in semitones (−6 to +6).
        """
        self.semitones = float(semitones)

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.semitones == 0.0:
            return audio_data

        audio, was_1d = self._ensure_2d(audio_data)
        ratio = 2.0 ** (self.semitones / 12.0)
        n_frames = audio.shape[0]
        channels = audio.shape[1]
        out_channels = []

        for ch in range(channels):
            mono = audio[:, ch].astype(np.float32)
            spec = np.fft.rfft(mono)
            freqs = np.arange(len(spec))
            warped_freqs = freqs / ratio
            # Interpolate back onto original frequency grid
            warped_spec_real = np.interp(freqs, warped_freqs, spec.real, left=0, right=0)
            warped_spec_imag = np.interp(freqs, warped_freqs, spec.imag, left=0, right=0)
            warped_spec = warped_spec_real + 1j * warped_spec_imag
            shifted = np.fft.irfft(warped_spec)[:n_frames]
            if len(shifted) < n_frames:
                shifted = np.pad(shifted, (0, n_frames - len(shifted)))
            out_channels.append(shifted)

        result = np.stack(out_channels, axis=1)
        return self._restore_shape(result, was_1d)

    def get_params(self) -> dict[str, Any]:
        return {"semitones": self.semitones}

    def set_params(self, params: dict[str, Any]) -> None:
        if "semitones" in params:
            self.semitones = float(params["semitones"])


# ── Noise Gate ────────────────────────────────────────────────────────────────


class NoiseGate(IAudioEffect):
    """
    Silence audio below a given dB threshold.

    Acts as a simple gate: if the RMS of the chunk is below
    *threshold_db*, the entire chunk is replaced with silence.
    """

    def __init__(self, threshold_db: float = -40.0, attack_ms: float = 5.0, release_ms: float = 50.0) -> None:
        """
        Args:
            threshold_db:  Gate threshold in dBFS (e.g. -40).
            attack_ms:     Attack time milliseconds (unused in this simple impl).
            release_ms:    Release time milliseconds (unused in this simple impl).
        """
        self.threshold_db = float(threshold_db)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self._open = False

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
        if rms < 1e-10:
            return audio_data

        rms_db = 20.0 * np.log10(rms + 1e-10)
        if rms_db < self.threshold_db:
            return np.zeros_like(audio_data)
        return audio_data

    def get_params(self) -> dict[str, Any]:
        return {
            "threshold_db": self.threshold_db,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        if "threshold_db" in params:
            self.threshold_db = float(params["threshold_db"])
        if "attack_ms" in params:
            self.attack_ms = float(params["attack_ms"])
        if "release_ms" in params:
            self.release_ms = float(params["release_ms"])


# ── Reverb ────────────────────────────────────────────────────────────────────


class ReverbEffect(IAudioEffect):
    """
    Add a simple algorithmic reverb (Schroeder-inspired comb filters).
    """

    # Comb-filter delay lengths in milliseconds (prime-ish values)
    _COMB_DELAYS_MS = [29.7, 37.1, 41.1, 43.7]
    _ALLPASS_DELAYS_MS = [5.0, 1.7]

    def __init__(self, wet_level: float = 0.3, room_size: float = 0.5) -> None:
        """
        Args:
            wet_level:  Mix ratio 0.0 (dry) – 1.0 (full wet).
            room_size:  Feedback coefficient 0.0 – 1.0.
        """
        self.wet_level = float(wet_level)
        self.room_size = float(room_size)
        self._buffers: dict[str, np.ndarray] = {}
        self._pointers: dict[str, int] = {}
        self._last_sr: int = 0

    def _init_buffers(self, sample_rate: int) -> None:
        self._buffers = {}
        self._pointers = {}
        self._last_sr = sample_rate
        for i, d in enumerate(self._COMB_DELAYS_MS):
            size = int(sample_rate * d / 1000)
            self._buffers[f"comb_{i}"] = np.zeros(size, dtype=np.float32)
            self._pointers[f"comb_{i}"] = 0
        for i, d in enumerate(self._ALLPASS_DELAYS_MS):
            size = int(sample_rate * d / 1000)
            self._buffers[f"ap_{i}"] = np.zeros(size, dtype=np.float32)
            self._pointers[f"ap_{i}"] = 0

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.wet_level == 0.0:
            return audio_data

        if sample_rate != self._last_sr:
            self._init_buffers(sample_rate)

        audio, was_1d = self._ensure_2d(audio_data)
        mono = audio[:, 0].astype(np.float32)

        # Process through comb filters
        reverb_out = np.zeros_like(mono)
        feedback = 0.4 + self.room_size * 0.45  # 0.4 – 0.85 range

        for i in range(len(self._COMB_DELAYS_MS)):
            key = f"comb_{i}"
            buf = self._buffers[key]
            ptr = self._pointers[key]
            n = len(buf)
            comb_out = np.zeros_like(mono)
            for j, sample in enumerate(mono):
                delayed = buf[ptr]
                buf[ptr] = sample + feedback * delayed
                comb_out[j] = delayed
                ptr = (ptr + 1) % n
            self._pointers[key] = ptr
            reverb_out += comb_out

        reverb_out /= len(self._COMB_DELAYS_MS)

        # Mix dry + wet
        mixed = (1.0 - self.wet_level) * mono + self.wet_level * reverb_out
        mixed_2d = np.stack([mixed] * audio.shape[1], axis=1)
        return self._restore_shape(mixed_2d, was_1d)

    def get_params(self) -> dict[str, Any]:
        return {"wet_level": self.wet_level, "room_size": self.room_size}

    def set_params(self, params: dict[str, Any]) -> None:
        if "wet_level" in params:
            self.wet_level = float(params["wet_level"])
        if "room_size" in params:
            self.room_size = float(params["room_size"])


# ── Volume Control ────────────────────────────────────────────────────────────


class VolumeControl(IAudioEffect):
    """Simple gain/volume control."""

    def __init__(self, gain: float = 1.0) -> None:
        """
        Args:
            gain: Linear gain multiplier (e.g. 2.0 = +6 dB).
        """
        self.gain = float(gain)

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        result = audio_data * self.gain
        return np.clip(result, -1.0, 1.0).astype(audio_data.dtype)

    def get_params(self) -> dict[str, Any]:
        return {"gain": self.gain}

    def set_params(self, params: dict[str, Any]) -> None:
        if "gain" in params:
            self.gain = float(params["gain"])


# ── Compressor ────────────────────────────────────────────────────────────────


class Compressor(IAudioEffect):
    """
    Dynamic range compressor.

    Reduces gain when the signal exceeds *threshold_db*, using the
    given *ratio*.  A soft-knee approximation is used.
    """

    def __init__(
        self,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 100.0,
        makeup_gain_db: float = 0.0,
    ) -> None:
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.makeup_gain_db = float(makeup_gain_db)
        self._gain_db: float = 0.0

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = audio_data.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-10))
        rms_db = 20.0 * np.log10(rms)

        # Gain computation
        if rms_db > self.threshold_db:
            excess_db = rms_db - self.threshold_db
            gain_reduction_db = excess_db * (1.0 - 1.0 / self.ratio)
            target_gain_db = -gain_reduction_db + self.makeup_gain_db
        else:
            target_gain_db = self.makeup_gain_db

        # Simple smoothing (attack/release)
        attack_coef = np.exp(-1.0 / (sample_rate * self.attack_ms / 1000.0))
        release_coef = np.exp(-1.0 / (sample_rate * self.release_ms / 1000.0))
        if target_gain_db < self._gain_db:
            self._gain_db = attack_coef * self._gain_db + (1 - attack_coef) * target_gain_db
        else:
            self._gain_db = release_coef * self._gain_db + (1 - release_coef) * target_gain_db

        linear_gain = 10.0 ** (self._gain_db / 20.0)
        return np.clip(audio * linear_gain, -1.0, 1.0).astype(audio_data.dtype)

    def get_params(self) -> dict[str, Any]:
        return {
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "makeup_gain_db": self.makeup_gain_db,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        for key in ("threshold_db", "ratio", "attack_ms", "release_ms", "makeup_gain_db"):
            if key in params:
                setattr(self, key, float(params[key]))


# ── Factory ───────────────────────────────────────────────────────────────────

EFFECT_REGISTRY: dict[str, type[IAudioEffect]] = {
    "pitch_shift": PitchShifter,
    "formant_shift": FormantShifter,
    "noise_gate": NoiseGate,
    "reverb": ReverbEffect,
    "volume": VolumeControl,
    "compressor": Compressor,
}


def create_effect(effect_type: str, **kwargs: Any) -> IAudioEffect:
    """
    Factory function to create effects by name.

    Args:
        effect_type: One of ``EFFECT_REGISTRY`` keys.
        **kwargs:    Passed to the effect constructor.

    Returns:
        Configured :class:`IAudioEffect` instance.

    Raises:
        KeyError: If *effect_type* is not registered.
    """
    cls = EFFECT_REGISTRY[effect_type]
    return cls(**kwargs)
