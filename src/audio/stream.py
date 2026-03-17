"""
Real-time audio streaming pipeline.

Connects AudioCapture → ProcessingPipeline → AudioOutput and manages
the full lifecycle: start, stop, pause, and monitoring.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

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
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        self._capture = AudioCapture(
            device_index=input_device,
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
        )
        self._output = AudioOutput(
            device_index=output_device,
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
        )

        self._processor: Optional[callable] = None
        self._paused = False
        self._running = False

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
        logger.info("AudioStream started.")

    def stop(self) -> None:
        """Stop the stream completely."""
        if not self._running:
            return
        self._capture.stop()
        self._output.stop()
        self._capture.remove_callback(self._on_audio_chunk)
        self._running = False
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
        """Called by AudioCapture on every new chunk (background thread)."""
        t_start = time.perf_counter()

        if self._paused:
            # Output silence
            silence = np.zeros_like(chunk)
            self._output.write(silence)
            return

        if self._processor is not None:
            try:
                processed = self._processor(chunk, self.sample_rate)
            except Exception as exc:
                logger.error("Processor error: %s", exc)
                processed = chunk  # pass-through on error
        else:
            processed = chunk  # pass-through mode

        self._output.write(processed)

        # Rolling latency estimate (EMA)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self._latency_ms = 0.9 * self._latency_ms + 0.1 * elapsed_ms
        self._chunk_count += 1
