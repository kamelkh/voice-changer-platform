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

# ── GPU availability check ────────────────────────────────────────────────────
try:
    import torch
    _CUDA = torch.cuda.is_available()
    if _CUDA:
        _DEVICE = torch.device("cuda")
        # Warm up CUDA so the first inference isn't slow
        _ = torch.zeros(1, device=_DEVICE)
        logger_tmp = None  # will log after logger is created
    else:
        _DEVICE = torch.device("cpu")
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    _CUDA = False
    _DEVICE = None

logger = get_logger(__name__)

if _TORCH_OK:
    if _CUDA:
        logger.info("PitchShifter: CUDA GPU detected (%s) — using GPU acceleration.",
                    torch.cuda.get_device_name(0))
    else:
        logger.info("PitchShifter: torch available but no CUDA GPU — using CPU.")


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
    Real-time pitch shifter.

    When PyTorch + CUDA is available (RTX 3070 etc.) uses a GPU phase-vocoder
    via torchaudio.functional.phase_vocoder for very low latency.
    Falls back to scipy.signal.resample on CPU otherwise.
    """

    def __init__(self, semitones: float = 0.0) -> None:
        self.semitones = float(semitones)
        self._buf = np.array([], dtype=np.float32)

        # Detect torchaudio for high-quality GPU pitch shift
        self._torchaudio_ok = False
        if _TORCH_OK and _CUDA:
            try:
                import torchaudio  # noqa: F401
                self._torchaudio_ok = True
            except ImportError:
                pass

    # GPU is only beneficial for chunks >= 4096 samples.
    # Below that the CUDA kernel-launch + CPU↔GPU copy overhead dominates.
    _GPU_MIN_SAMPLES = 4096

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.semitones == 0.0:
            return audio_data

        mono, was_2d, channels = self._to_mono_1d(audio_data)

        if (self._torchaudio_ok and _CUDA
                and len(mono) >= self._GPU_MIN_SAMPLES):
            out = self._process_gpu(mono, sample_rate)
        else:
            out = self._process_cpu(mono)

        return self._restore(out, was_2d, channels)

    def _process_gpu(self, mono: np.ndarray, sample_rate: int) -> np.ndarray:
        """GPU pitch shift using torchaudio.functional.resample + phase trick."""
        import torchaudio.functional as F  # noqa

        ratio = 2.0 ** (self.semitones / 12.0)
        orig_sr = sample_rate
        # Resample to a "wrong" sample rate then resample back at original rate
        # (same as time-stretch + resample = pitch shift without duration change)
        shifted_sr = int(round(orig_sr / ratio))

        t = torch.from_numpy(mono).unsqueeze(0).to(_DEVICE)  # [1, T]
        # Step 1: time-stretch by resampling to shifted_sr
        stretched = F.resample(t, orig_freq=orig_sr, new_freq=shifted_sr)
        # Step 2: resample back to orig_sr to preserve duration
        out_t = F.resample(stretched, orig_freq=shifted_sr, new_freq=orig_sr)

        out = out_t.squeeze(0).cpu().numpy().astype(np.float32)
        # Trim/pad to match input length exactly
        n = len(mono)
        if len(out) >= n:
            return out[:n]
        return np.concatenate([out, np.zeros(n - len(out), dtype=np.float32)])

    def _process_cpu(self, mono: np.ndarray) -> np.ndarray:
        """CPU pitch shift using linear interpolation (no FFT leakage).

        Unlike scipy.signal.resample, linear interpolation does not
        introduce spectral-leakage artefacts at chunk boundaries,
        which eliminates the robotic / metallic sound on small chunks.
        """
        n = len(mono)
        ratio = 2.0 ** (self.semitones / 12.0)
        # Read the input at a different speed, then take exactly n samples.
        # ratio > 1 → read faster → higher pitch; ratio < 1 → slower → lower.
        src_indices = np.arange(n, dtype=np.float32) / ratio
        src_indices = np.clip(src_indices, 0, n - 1)
        return np.interp(src_indices, np.arange(n, dtype=np.float32), mono).astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {"semitones": self.semitones}

    def set_params(self, params: dict[str, Any]) -> None:
        if "semitones" in params:
            self.semitones = float(params["semitones"])
            # Re-check torchaudio if parameters change



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
        n = len(mono)
        if n < 16:
            return audio_data

        # Measure input RMS for preservation
        in_rms = float(np.sqrt(np.mean(mono ** 2))) + 1e-10

        ratio = 2.0 ** (self.semitones / 12.0)

        spec = np.fft.rfft(mono)
        freqs = np.arange(len(spec), dtype=np.float32)
        warped = freqs / ratio

        # Use edge-value extrapolation (no left/right=0 zeroing)
        real = np.interp(freqs, warped, spec.real)
        imag = np.interp(freqs, warped, spec.imag)
        out = np.fft.irfft(real + 1j * imag)[:n].astype(np.float32)
        if len(out) < n:
            out = np.pad(out, (0, n - len(out)))

        # Preserve RMS level
        out_rms = float(np.sqrt(np.mean(out ** 2))) + 1e-10
        out *= (in_rms / out_rms)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

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

    def __init__(self, threshold_db: float = -50.0,
                 attack_ms: float = 2.0, release_ms: float = 150.0) -> None:
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


# ── Voice Disguise ─────────────────────────────────────────────────────────


class VoiceDisguise(IAudioEffect):
    """
    Identity-masking effect that makes a voice unrecognisable.

    Combines three techniques:
    1. **Spectral noise injection** – band-limited noise shaped by the
       signal envelope, breaking unique vocal fingerprints.
    2. **Micro-pitch modulation** – slow vibrato (3-6 Hz) that shifts
       the fundamental frequency irregularly.
    3. **Formant smearing** – slight random warping of the spectral
       envelope each chunk, destroying consistent formant patterns.

    *intensity* ranges from 0.0 (off) to 1.0 (maximum disguise).
    """

    def __init__(self, intensity: float = 0.5) -> None:
        self.intensity = float(intensity)
        self._phase: float = 0.0
        # Pre-computed bandpass filter coefficients (set on first call)
        self._bp_b: np.ndarray | None = None
        self._bp_a: np.ndarray | None = None
        self._bp_sr: int = 0
        self._noise_state: np.ndarray | None = None

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.intensity == 0.0:
            return audio_data

        mono, was_2d, channels = self._to_mono_1d(audio_data)
        n = len(mono)
        if n < 64:
            return audio_data

        # Measure input RMS for preservation
        in_rms = float(np.sqrt(np.mean(mono ** 2))) + 1e-10

        # ── 1. Band-limited spectral noise (300–3000 Hz) ─────────────────
        # Very light noise — the formant smearing and micro-pitch
        # modulation do the real disguising work.  Keep noise minimal
        # to avoid audible hiss.
        noise_factor = 0.008
        noise = np.random.randn(n).astype(np.float32) * in_rms * self.intensity * noise_factor

        # Build bandpass filter once per sample-rate
        if self._bp_b is None or self._bp_sr != sample_rate:
            nyq = sample_rate / 2.0
            lo = min(300 / nyq, 0.98)
            hi = min(3000 / nyq, 0.98)
            if lo < hi:
                self._bp_b, self._bp_a = signal.butter(3, [lo, hi], btype="band")
            else:
                self._bp_b, self._bp_a = np.array([1.0]), np.array([1.0])
            self._bp_sr = sample_rate
            self._noise_state = None

        # Apply bandpass with state continuity between chunks
        if self._noise_state is None:
            self._noise_state = signal.lfiltic(self._bp_b, self._bp_a, [])

        noise_filt, self._noise_state = signal.lfilter(
            self._bp_b, self._bp_a, noise, zi=self._noise_state
        )

        # ── 2. Micro-pitch modulation (irregular vibrato) ────────────────
        t = np.arange(n, dtype=np.float32) / sample_rate + self._phase
        mod_freq = 2.0 + self.intensity * 1.5          # 2.0 – 3.5 Hz (gentle)
        mod_depth = 0.0005 + self.intensity * 0.0015     # very subtle shift
        phase_offset = mod_depth * np.sin(2.0 * np.pi * mod_freq * t)
        # Bound phase to prevent float overflow after long sessions
        self._phase = (self._phase + n / sample_rate) % 100.0

        indices = np.arange(n, dtype=np.float32) + phase_offset * sample_rate * 0.001
        indices = np.clip(indices, 0, n - 1)
        modulated = np.interp(np.arange(n, dtype=np.float32), indices, mono).astype(np.float32)

        # ── 3. Formant smear – random spectral stretch per chunk ─────────
        spec = np.fft.rfft(modulated)
        n_bins = len(spec)
        # Slight random warp factor that changes each chunk
        warp = 1.0 + (np.random.rand() - 0.5) * 0.03 * self.intensity
        old_bins = np.arange(n_bins, dtype=np.float32)
        new_bins = old_bins / warp
        # Edge-value extrapolation (no left=0/right=0 zeroing)
        spec_real = np.interp(old_bins, new_bins, spec.real)
        spec_imag = np.interp(old_bins, new_bins, spec.imag)
        smeared = np.fft.irfft(spec_real + 1j * spec_imag)[:n].astype(np.float32)

        if len(smeared) < n:
            smeared = np.pad(smeared, (0, n - len(smeared)))

        # ── Combine ──────────────────────────────────────────────────────
        out = smeared + noise_filt.astype(np.float32)

        # ── RMS preservation ─────────────────────────────────────────────
        out_rms = float(np.sqrt(np.mean(out ** 2))) + 1e-10
        out *= (in_rms / out_rms)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        return self._restore(out, was_2d, channels)

    def get_params(self) -> dict[str, Any]:
        return {"intensity": self.intensity}

    def set_params(self, params: dict[str, Any]) -> None:
        if "intensity" in params:
            self.intensity = float(params["intensity"])


# ── Factory ───────────────────────────────────────────────────────────────

EFFECT_REGISTRY: dict[str, type[IAudioEffect]] = {
    "pitch_shift":     PitchShifter,
    "formant_shift":   FormantShifter,
    "noise_gate":      NoiseGate,
    "reverb":          ReverbEffect,
    "volume":          VolumeControl,
    "compressor":      Compressor,
    "voice_disguise":  VoiceDisguise,
}


def create_effect(effect_type: str, **kwargs: Any) -> IAudioEffect:
    """Factory: create an effect by registry name."""
    return EFFECT_REGISTRY[effect_type](**kwargs)

