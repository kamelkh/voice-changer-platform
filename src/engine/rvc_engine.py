"""
RVC (Retrieval-based Voice Conversion) inference engine.

Supports RVC v2 models loaded from .pth files.
GPU acceleration via CUDA; falls back to CPU automatically.

Uses:
  - HuBERT (via ``transformers``) for content feature extraction
  - pyworld / harvest for F0 extraction
  - SynthesizerTrnMs768NSFsid (src.engine.rvc_models) for voice synthesis
  - Optional FAISS index for feature retrieval
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.constants import (
    MODELS_DIR,
    RVC_DEFAULT_F0_METHOD,
    RVC_DEFAULT_FILTER_RADIUS,
    RVC_DEFAULT_INDEX_RATE,
    RVC_DEFAULT_PROTECT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RVCEngine:
    """
    RVC v2 voice conversion engine with real inference.

    Uses HuBERT (transformers) for feature extraction, pyworld for F0,
    and SynthesizerTrnMs768NSFsid for voice synthesis.

    Example::

        engine = RVCEngine()
        engine.load_model("models/my_voice.pth")
        converted = engine.convert(audio_chunk, sample_rate=48000)
    """

    # HuBERT output rate: 1 frame per 320 samples at 16 kHz (= 50 fps)
    _HUBERT_HOP = 320
    _HUBERT_SR = 16000

    def __init__(
        self,
        use_gpu: bool = True,
        f0_method: str = RVC_DEFAULT_F0_METHOD,
        pitch_shift: int = 0,
        index_rate: float = RVC_DEFAULT_INDEX_RATE,
        filter_radius: int = RVC_DEFAULT_FILTER_RADIUS,
        protect: float = RVC_DEFAULT_PROTECT,
    ) -> None:
        self.f0_method = f0_method
        self.pitch_shift = pitch_shift
        self.index_rate = index_rate
        self.filter_radius = filter_radius
        self.protect = protect

        self._model_path: Optional[Path] = None
        self._index_path: Optional[Path] = None
        self._model_loaded: bool = False
        self._model_sr: int = 40000  # RVC model output sample rate

        # Determine device
        self._device: str = "cpu"
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info("RVC engine using CUDA: %s", torch.cuda.get_device_name(0))
                else:
                    logger.info("CUDA not available — RVC engine using CPU.")
            except ImportError:
                logger.warning("torch not installed — RVC engine using CPU.")

        # Lazy components
        self._net_g = None        # SynthesizerTrnMs768NSFsid
        self._cpt: Optional[dict] = None
        self._index = None        # FAISS index
        self._big_npy = None      # Pre-reconstructed FAISS vectors
        self._hubert_model = None # HuBERT feature extractor
        self._hubert_processor = None

        # Audio buffer for accumulating short chunks before conversion.
        # F0 extraction and HuBERT need ≥0.3 s of context for quality.
        self._MIN_BUFFER_SAMPLES = 4800  # ~0.3 s at 16 kHz
        self._audio_buf: list[np.ndarray] = []
        self._buf_samples: int = 0
        self._output_queue: list[np.ndarray] = []  # pre-converted chunks

        # Output sample rate — when set (by pipeline), the engine resamples
        # the model output directly to this rate instead of back to the
        # input rate.  Avoids the lossy input_sr→model_sr→input_sr round-trip.
        self._output_sr: Optional[int] = None

    # ── Model management ──────────────────────────────────────────────────────

    def load_model(self, model_path: str | Path,
                   index_path: Optional[str | Path] = None) -> bool:
        """Load an RVC v2 .pth model.  Returns True on success."""
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            return False

        try:
            import torch
            from src.engine.rvc_models import build_model_from_checkpoint

            logger.info("Loading RVC model: %s …", model_path.name)
            self._cpt = torch.load(model_path, map_location=self._device,
                                   weights_only=False)
            self._model_path = model_path

            # Parse model sample rate from checkpoint
            sr_str = self._cpt.get("sr", "40k")
            self._model_sr = int(str(sr_str).replace("k", "000"))

            # Build the real synthesizer network
            self._net_g = build_model_from_checkpoint(self._cpt, self._device)
            logger.info("RVC network built (%s, sr=%d).",
                        self._cpt.get("version", "?"), self._model_sr)

            # Index
            if index_path is not None:
                self._load_index(Path(index_path))
            else:
                self._auto_discover_index(model_path)

            # HuBERT
            self._load_hubert()

            self._audio_buf.clear()
            self._buf_samples = 0
            self._output_queue.clear()

            self._model_loaded = True
            logger.info("RVC model ready: %s (device=%s)",
                        model_path.name, self._device)
            return True

        except Exception as exc:
            logger.error("Failed to load RVC model '%s': %s", model_path, exc)
            return False

    def unload_model(self) -> None:
        """Release the loaded model and free GPU memory."""
        self._net_g = None
        self._index = None
        self._big_npy = None
        self._cpt = None
        self._model_loaded = False
        self._model_path = None
        self._audio_buf.clear()
        self._buf_samples = 0
        self._output_queue.clear()

        try:
            import torch
            if self._device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("RVC model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_name(self) -> str:
        if self._model_path:
            return self._model_path.stem
        return ""

    # ── Inference ─────────────────────────────────────────────────────────────

    def convert(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Convert audio through the loaded RVC model.

        Buffers short chunks internally and converts when enough audio
        has accumulated (≥0.3 s) for reliable F0 extraction.  Returns
        converted audio matched to the input chunk length (or scaled
        to ``_output_sr`` when the pipeline has set a target output rate).
        """
        if not self._model_loaded or self._net_g is None:
            return audio_data

        mono = audio_data.flatten().astype(np.float32)
        chunk_len = len(mono)

        # Determine the actual output rate and corresponding chunk length
        out_sr = self._output_sr if self._output_sr else sample_rate
        out_chunk_len = int(round(chunk_len * out_sr / sample_rate)) if out_sr != sample_rate else chunk_len

        # Always accumulate incoming audio so nothing is dropped
        self._audio_buf.append(mono)
        self._buf_samples += chunk_len

        # If there is pre-converted audio waiting, serve from queue
        if self._output_queue:
            return self._dequeue_chunk(out_chunk_len)

        # Not enough audio yet — pass through (resampled to output rate)
        if self._buf_samples < self._MIN_BUFFER_SAMPLES:
            if out_sr != sample_rate:
                import torch
                import torchaudio.functional as AF
                t = torch.from_numpy(mono).float().unsqueeze(0)
                t = AF.resample(t, sample_rate, out_sr)
                return t.squeeze(0).numpy().astype(np.float32)
            return audio_data

        # ── Convert the full buffer ───────────────────────────────────
        full_audio = np.concatenate(self._audio_buf)
        n_chunks = len(self._audio_buf)
        self._audio_buf.clear()
        self._buf_samples = 0

        try:
            converted = self._convert_block(full_audio, sample_rate, out_sr)
        except Exception as exc:
            logger.error("RVC inference error: %s", exc)
            converted = full_audio

        # Split converted audio back into chunk-sized pieces
        offset = 0
        pieces = []
        for _ in range(n_chunks):
            end = min(offset + out_chunk_len, len(converted))
            piece = converted[offset:end]
            if len(piece) < out_chunk_len:
                piece = np.pad(piece, (0, out_chunk_len - len(piece)))
            pieces.append(piece)
            offset += out_chunk_len

        # Return first chunk, queue the rest
        self._output_queue.extend(pieces[1:])
        return pieces[0].astype(np.float32)

    def _dequeue_chunk(self, chunk_len: int) -> np.ndarray:
        """Return one buffered chunk from the output queue."""
        chunk = self._output_queue.pop(0)
        if len(chunk) != chunk_len:
            chunk = chunk[:chunk_len] if len(chunk) > chunk_len else \
                    np.pad(chunk, (0, chunk_len - len(chunk)))
        return chunk.astype(np.float32)

    def _convert_block(self, audio: np.ndarray, sample_rate: int,
                       output_sample_rate: int) -> np.ndarray:
        """Run the full RVC conversion pipeline on a block of audio."""
        import torch
        import torch.nn.functional as F_t
        import torchaudio.functional as AF

        orig_len = len(audio)
        # Expected output length at the target output rate
        out_len = int(round(orig_len * output_sample_rate / sample_rate)) if output_sample_rate != sample_rate else orig_len

        # 1. Resample to 16 kHz for HuBERT
        if sample_rate != self._HUBERT_SR:
            t = torch.from_numpy(audio).unsqueeze(0)
            t = AF.resample(t, sample_rate, self._HUBERT_SR)
            mono_16k = t.squeeze(0).numpy()
        else:
            mono_16k = audio

        # 1b. High-pass filter full audio (matches reference pipeline)
        from scipy.signal import butter, filtfilt
        bh, ah = butter(N=5, Wn=48, btype="high", fs=16000)
        mono_16k = filtfilt(bh, ah, mono_16k).astype(np.float32)

        # 1c. Reflection-pad for boundary quality (0.3 s each side)
        pad_16k = 4800  # 0.3 seconds at 16 kHz
        mono_16k_padded = np.pad(mono_16k, (pad_16k, pad_16k), mode="reflect")

        # 2. Extract HuBERT features (768-dim) — output at 50 fps
        feats = self._extract_features(mono_16k_padded, apply_hp=False)
        if feats is None:
            return audio

        # 3. Upsample features 2× to match model frame rate (100 fps)
        #    Reference: F.interpolate(feats.permute(0,2,1), scale_factor=2)
        feats_t = torch.from_numpy(feats).permute(0, 2, 1)  # (1, 768, T)
        feats_t = F_t.interpolate(feats_t, scale_factor=2, mode='nearest')
        feats = feats_t.permute(0, 2, 1).numpy()  # (1, T*2, 768)

        # p_len: number of frames at 100 fps (window=160 at 16 kHz)
        p_len = feats.shape[1]

        # 4. Extract F0 at 100 fps (natural frame_period=10 ms resolution)
        f0 = self._extract_f0(mono_16k_padded, self._HUBERT_SR, p_len)
        pitch = self._f0_to_pitch_index(f0)

        # 5. Feature retrieval (save original for protect)
        feats0 = feats.copy() if self.protect < 0.5 else None
        if self._index is not None and self.index_rate > 0:
            feats = self._index_retrieval(feats)

        # 6. Consonant protection — blend back original feats in unvoiced regions
        if self.protect < 0.5 and feats0 is not None:
            feats = self._apply_protect(feats, feats0, f0)

        # 7. Run generator
        with torch.no_grad():
            phone_t = torch.from_numpy(feats).to(self._device)
            pitch_t = torch.from_numpy(pitch).long().to(self._device)
            pitchf_t = torch.from_numpy(f0).float().to(self._device)
            sid_t = torch.tensor([0], device=self._device)
            lengths_t = torch.tensor([p_len], device=self._device)

            audio_out = self._net_g(
                phone_t, lengths_t, pitch_t, pitchf_t, sid_t
            )

        audio_out = audio_out.squeeze().cpu().numpy().astype(np.float32)

        # 8. Trim padding at model sample rate BEFORE resampling
        #    (Reference trims at tgt_sr before any resample)
        pad_model = int(pad_16k * self._model_sr / self._HUBERT_SR)
        if len(audio_out) > pad_model * 2:
            audio_out = audio_out[pad_model:-pad_model]

        # 9. Resample from model_sr directly to output rate
        #    (avoids the lossy model_sr→input_sr→output_sr round-trip)
        if output_sample_rate != self._model_sr:
            t_out = torch.from_numpy(audio_out).unsqueeze(0)
            t_out = AF.resample(t_out, self._model_sr, output_sample_rate)
            audio_out = t_out.squeeze(0).numpy()

        # 10. Match expected output length
        if len(audio_out) >= out_len:
            return audio_out[:out_len]
        return np.pad(audio_out, (0, out_len - len(audio_out)))

    # ── Private: feature extraction ───────────────────────────────────────────

    def _load_hubert(self) -> None:
        """Load contentvec HuBERT model used by RVC training."""
        if self._hubert_model is not None:
            return  # already loaded

        try:
            import torch
            from transformers import HubertModel
            from huggingface_hub import hf_hub_download

            # contentvec768l12 — the fine-tuned HuBERT used to train RVC models
            model_id = "lengyue233/content-vec-best"
            logger.info("Loading contentvec HuBERT (%s) …", model_id)
            model = HubertModel.from_pretrained(model_id)

            # Fix: weight_norm format changed in newer PyTorch.
            # The checkpoint has weight_g/weight_v but the model expects
            # parametrizations.weight.original0/original1.  Load them manually.
            raw_path = hf_hub_download(model_id, "pytorch_model.bin")
            raw_sd = torch.load(raw_path, map_location="cpu", weights_only=True)
            conv = model.encoder.pos_conv_embed.conv
            if hasattr(conv, "parametrizations"):
                wg = raw_sd.get("encoder.pos_conv_embed.conv.weight_g")
                wv = raw_sd.get("encoder.pos_conv_embed.conv.weight_v")
                if wg is not None and wv is not None:
                    conv.parametrizations.weight.original0.data.copy_(wg)
                    conv.parametrizations.weight.original1.data.copy_(wv)
                    logger.info("Fixed pos_conv_embed weight_norm weights.")
            del raw_sd

            model.eval()
            model.to(self._device)
            self._hubert_model = model
            logger.info("HuBERT loaded on %s.", self._device)
        except Exception as exc:
            logger.error("Failed to load HuBERT: %s", exc)

    def _extract_features(self, audio_16k: np.ndarray,
                          apply_hp: bool = True) -> Optional[np.ndarray]:
        """Extract 768-dim HuBERT features.  Returns (1, T, 768) ndarray."""
        try:
            import torch

            if self._hubert_model is None:
                n_frames = max(1, len(audio_16k) // self._HUBERT_HOP)
                logger.warning("HuBERT not loaded — using zero features.")
                return np.zeros((1, n_frames, 768), dtype=np.float32)

            if apply_hp:
                from scipy.signal import butter, lfilter
                bh, ah = butter(N=5, Wn=48, btype="high", fs=16000)
                audio_16k = lfilter(bh, ah, audio_16k).astype(np.float32)

            with torch.no_grad():
                t = torch.from_numpy(audio_16k).unsqueeze(0).float().to(self._device)
                out = self._hubert_model(t, output_hidden_states=True)
                feats = out.last_hidden_state  # (1, T, 768)

            return feats.cpu().numpy().astype(np.float32)

        except Exception as exc:
            logger.error("HuBERT feature extraction error: %s", exc)
            return None

    # ── Private: F0 extraction ────────────────────────────────────────────────

    def _extract_f0(self, audio_16k: np.ndarray, sr: int,
                    n_frames: int) -> np.ndarray:
        """Extract F0 contour aligned to HuBERT frame count.  Returns (1, T)."""
        try:
            import pyworld
            from scipy.signal import medfilt

            # harvest gives one F0 per 10 ms frame_period
            f0, t = pyworld.harvest(
                audio_16k.astype(np.float64), sr,
                f0_floor=50, f0_ceil=1100, frame_period=10,
            )
            f0 = pyworld.stonemask(audio_16k.astype(np.float64), f0, t, sr)

            # Resample F0 to match HuBERT frame count
            if len(f0) != n_frames:
                x_old = np.linspace(0, 1, len(f0))
                x_new = np.linspace(0, 1, n_frames)
                f0 = np.interp(x_new, x_old, f0)

            # Apply pitch shift
            if self.pitch_shift != 0:
                voiced = f0 > 1.0
                f0[voiced] = f0[voiced] * (2.0 ** (self.pitch_shift / 12.0))

            # Median filter for smoothing
            if self.filter_radius > 0 and len(f0) > self.filter_radius * 2 + 1:
                f0 = medfilt(f0, kernel_size=self.filter_radius * 2 + 1)

            return f0.astype(np.float32).reshape(1, -1)  # (1, T)

        except Exception as exc:
            logger.error("F0 extraction error: %s", exc)
            return np.zeros((1, n_frames), dtype=np.float32)

    @staticmethod
    def _f0_to_pitch_index(f0: np.ndarray) -> np.ndarray:
        """Convert continuous F0 (Hz) to pitch embedding index (0–255).

        Uses mel-scale mapping matching the RVC reference:
          f0_mel = 1127 * log(1 + f0/700)
          index  = (f0_mel - mel_min) * 254 / (mel_max - mel_min) + 1
        Unvoiced (f0<=0) maps to 1.  Returns (1, T) int array.
        """
        f0_mel_min = 1127 * np.log(1 + 50.0 / 700)     # ~f0_floor  50 Hz
        f0_mel_max = 1127 * np.log(1 + 1100.0 / 700)    # ~f0_ceil 1100 Hz

        f0_flat = f0.flatten().copy()
        f0_mel = 1127.0 * np.log(1 + f0_flat / 700.0)
        idx = np.ones_like(f0_flat, dtype=np.int64)
        voiced = f0_mel > 0
        idx[voiced] = np.clip(
            np.round(
                (f0_mel[voiced] - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1
            ).astype(np.int64),
            1, 255,
        )
        return idx.reshape(f0.shape)

    # ── Private: FAISS index retrieval ────────────────────────────────────────

    def _auto_discover_index(self, model_path: Path) -> None:
        """Look for a matching .index file in the models directory."""
        stem = model_path.stem.lower()
        models_dir = model_path.parent

        # Try exact match first
        exact = model_path.with_suffix(".index")
        if exact.exists():
            self._load_index(exact)
            return

        # Search for index files containing the model name
        for idx_file in models_dir.glob("*.index"):
            if stem in idx_file.stem.lower():
                self._load_index(idx_file)
                return

        logger.info("No FAISS index found for %s — feature retrieval disabled.",
                    model_path.name)

    def _load_index(self, index_path: Path) -> None:
        """Load the FAISS feature retrieval index."""
        try:
            import faiss
            self._index = faiss.read_index(str(index_path))
            self._index_path = index_path
            # Pre-reconstruct all vectors for fast retrieval (matches reference)
            self._big_npy = self._index.reconstruct_n(0, self._index.ntotal)
            logger.info("FAISS index loaded: %s (%d vectors)",
                        index_path.name, self._index.ntotal)
        except ImportError:
            logger.warning("faiss not installed — feature retrieval disabled.")
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)

    def _index_retrieval(self, feats: np.ndarray) -> np.ndarray:
        """Blend HuBERT features with FAISS nearest-neighbour features.

        Uses k=8 neighbours with inverse-distance-squared weighting,
        matching the reference RVC pipeline.
        """
        if self._index is None or self._big_npy is None:
            return feats

        try:
            npy = feats[0].copy()  # (T, 768)

            score, ix = self._index.search(npy, k=8)

            # Inverse-distance-squared weighting
            weight = np.square(1.0 / np.maximum(score, 1e-6))
            weight /= weight.sum(axis=1, keepdims=True)

            # Weighted combination of retrieved vectors
            npy_retrieved = np.sum(
                self._big_npy[ix] * np.expand_dims(weight, axis=2),
                axis=1,
            )

            # Blend with original features
            blended = (
                npy_retrieved * self.index_rate
                + npy * (1 - self.index_rate)
            )

            return blended.reshape(feats.shape).astype(np.float32)

        except Exception as exc:
            logger.warning("Index retrieval failed: %s", exc)
            return feats

    def _apply_protect(self, feats: np.ndarray, feats0: np.ndarray,
                       f0: np.ndarray) -> np.ndarray:
        """Protect consonants by blending original features in unvoiced regions.

        Matches the reference RVC pipeline: voiced regions keep the
        retrieval-modified features, unvoiced regions blend back to original.
        """
        pitchff = f0.flatten().copy()
        pitchff[pitchff > 0] = 1.0
        pitchff[pitchff < 1] = self.protect
        pitchff = pitchff.reshape(1, -1, 1)  # (1, T, 1)
        feats = feats * pitchff + feats0 * (1.0 - pitchff)
        return feats.astype(np.float32)

    # ── Parameter updates ─────────────────────────────────────────────────────

    def set_params(
        self,
        pitch_shift: Optional[int] = None,
        index_rate: Optional[float] = None,
        protect: Optional[float] = None,
        f0_method: Optional[str] = None,
    ) -> None:
        """Update inference parameters (can be called while running)."""
        if pitch_shift is not None:
            self.pitch_shift = int(pitch_shift)
        if index_rate is not None:
            self.index_rate = float(index_rate)
        if protect is not None:
            self.protect = float(protect)
        if f0_method is not None:
            self.f0_method = f0_method
