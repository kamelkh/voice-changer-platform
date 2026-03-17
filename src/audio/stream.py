"""
Real-time audio streaming pipeline.

Connects AudioCapture → ProcessingPipeline → AudioOutput and manages
the full lifecycle: start, stop, pause, and monitoring.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
from scipy.signal import resample_poly
from math import gcd

from src.audio.capture import AudioCapture
from src.audio.output import AudioOutput
from src.utils.constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CHANNELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLE_RATE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioStream:
    """
    Orchestrate the capure-process-output loop.

    The processing callback transforms each captured chunk and routes
    it to the output.  If no processing callback is set the stream
    operates in *pass-through* mode (raw mic → virtual mic).

    Example::

        stream = AudioStream(
            input_device=0,
            output_device=vb_cable_index,
        )
        stream.set_processor(pipeline.process)
        stream.start()
        ...
        stream.stop()
    """

    def __init__(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        input_sample_rate: Optional[int] = None,
        output_sample_rate: Optional[int] = None,
    ) -> None:
        self.input_sample_rate  = input_sample_rate  or sample_rate
        self.output_sample_rate = output_sample_rate or sample_rate
        self.sample_rate = self.input_sample_rate   # kept for back-compat
        self.channels = channels
        self.chunk_size = chunk_size

        # Pre-compute resample ratio (GCD-reduced)
        _g = gcd(self.output_sample_rate, self.input_sample_rate)
        self._rs_up   = self.output_sample_rate // _g
        self._rs_down = self.input_sample_rate  // _g

        self._capture = AudioCapture(
            device_index=input_device,
            sample_rate=self.input_sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
        )
        self._output = AudioOutput(
            device_index=output_device,
            sample_rate=self.output_sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
        )

        self._processor: Optional[callable] = None
        self._paused = False
        self._running = False

        # Dedicated processing thread + input queue (decouples capture ↔ process
        # so the PortAudio callback is never blocked by heavy DSP work)
        self._proc_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._proc_thread: Optional[threading.Thread] = None

        # Latency monitoring
        self._last_chunk_time: float = 0.0
        self._latency_ms: float = 0.0
        self._chunk_count: int = 0

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_processor(self, processor: callable) -> None:
        """
        Set the audio processing callback.

        The callable must accept a ``numpy.ndarray`` and return a
        processed ``numpy.ndarray`` of the same shape.

        Args:
            processor: Function ``(np.ndarray) -> np.ndarray``.
        """
        self._processor = processor

    def set_input_device(self, device_index: Optional[int]) -> None:
        """Change the input device.  Restarts the stream if running."""
        logger.info("[STREAM] set_input_device → %s (running=%s)", device_index, self._running)
        was_running = self._running
        if was_running:
            self.stop()
        self._capture.device_index = device_index
        if was_running:
            self.start()

    def set_output_device(self, device_index: Optional[int]) -> None:
        """Change the output device.  Restarts the stream if running."""
        was_running = self._running
        if was_running:
            self.stop()
        self._output.device_index = device_index
        if was_running:
            self.start()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start capturing, processing, and outputting audio."""
        if self._running:
            logger.warning("AudioStream already running.")
            return

        self._capture.add_callback(self._on_audio_chunk)
        self._output.start()
        self._capture.start()
        self._running = True
        self._paused = False
        # Start the dedicated processing thread
        self._proc_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="AudioProcThread"
        )
        self._proc_thread.start()
        logger.info("AudioStream started.")

    def stop(self) -> None:
        """Stop the stream completely."""
        if not self._running:
            return
        self._running = False
        self._capture.stop()
        self._capture.remove_callback(self._on_audio_chunk)
        # Unblock the processing thread so it can exit
        try:
            self._proc_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._proc_thread is not None:
            self._proc_thread.join(timeout=2.0)
            self._proc_thread = None
        self._output.stop()
        logger.info("AudioStream stopped.")

    def pause(self) -> None:
        """Pause processing (output silence while keeping streams open)."""
        self._paused = True
        logger.info("AudioStream paused.")

    def resume(self) -> None:
        """Resume processing after a pause."""
        self._paused = False
        logger.info("AudioStream resumed.")

    def toggle(self) -> bool:
        """Toggle between running and stopped states.

        Returns:
            *True* if the stream is now running.
        """
        if self._running:
            self.stop()
            return False
        else:
            self.start()
            return True

    # ── Monitoring ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True when the stream is active."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """True when the stream is paused."""
        return self._paused

    @property
    def latency_ms(self) -> float:
        """Approximate processing latency in milliseconds."""
        return self._latency_ms

    @property
    def input_level(self) -> float:
        """Current input RMS level (0.0 – 1.0)."""
        return self._capture.input_level

    @property
    def output_level(self) -> float:
        """Current output RMS level (0.0 – 1.0)."""
        return self._output.output_level

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Called by AudioCapture on every new chunk (background thread).
        This method must return as fast as possible – it only enqueues."""
        if not self._running:
            return
        try:
            self._proc_queue.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest chunk to make room for the new one
            try:
                self._proc_queue.get_nowait()
                self._proc_queue.put_nowait(chunk)
            except queue.Empty:
                pass

    def _write_resampled(self, audio: np.ndarray) -> None:
        """Resample *audio* and split into output-chunk-sized pieces."""
        resampled = self._maybe_resample(audio)
        out_cs = self._output.chunk_size
        n_frames = resampled.shape[0] if resampled.ndim > 1 else len(resampled)
        if n_frames <= out_cs:
            self._output.write(resampled)
        else:
            for i in range(0, n_frames, out_cs):
                piece = resampled[i : i + out_cs]
                self._output.write(piece)

    def _processing_loop(self) -> None:
        """Dedicated thread: dequeue → process → write to output."""
        while self._running:
            try:
                chunk = self._proc_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:  # sentinel – exit signal
                break

            t_start = time.perf_counter()

            if self._paused:
                silence = np.zeros_like(chunk)
                self._write_resampled(silence)
            elif self._processor is not None:
                try:
                    processed = self._processor(chunk, self.input_sample_rate)
                except Exception as exc:
                    logger.error("Processor error: %s", exc)
                    processed = chunk
                self._write_resampled(processed)
            else:
                self._write_resampled(chunk)

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self._latency_ms = 0.9 * self._latency_ms + 0.1 * elapsed_ms
            self._chunk_count += 1

            # Diagnostic: log every 200 chunks (~3 sec at 256/16kHz)
            if self._chunk_count % 200 == 1:
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                logger.info(
                    "[STREAM] chunk #%d  in_rms=%.6f  latency=%.1fms  q=%d",
                    self._chunk_count, rms, self._latency_ms,
                    self._proc_queue.qsize(),
                )

    def _maybe_resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio if input/output sample rates differ."""
        if self._rs_up == self._rs_down:
            return audio
        mono = audio.ndim == 1
        if mono:
            audio = audio.reshape(-1, 1)
        channels = audio.shape[1]
        out_cols = []
        for ch in range(channels):
            col = resample_poly(audio[:, ch].astype(np.float32),
                                self._rs_up, self._rs_down)
            out_cols.append(col)
        result = np.stack(out_cols, axis=1).astype(np.float32)
        return result[:, 0] if mono else result
