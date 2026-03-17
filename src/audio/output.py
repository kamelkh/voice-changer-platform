"""
Audio output module – writes processed audio to VB-Audio Virtual Cable.

The virtual cable's *CABLE Input* device appears as a PortAudio output.
Applications such as Zoom/WhatsApp Web/Discord read from the matching
*CABLE Output* device, which acts as a microphone.
"""
from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from src.utils.constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CHANNELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DTYPE,
    DEFAULT_SAMPLE_RATE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioOutput:
    """
    Write processed audio to an output device (typically VB-Cable).

    Audio chunks are placed in a thread-safe queue; the OutputStream
    callback drains that queue.  This decouples the processing thread
    from the PortAudio callback thread and prevents glitches.

    Example::

        output = AudioOutput(device_index=vb_cable_index)
        output.start()
        output.write(processed_chunk)  # call from processing thread
        ...
        output.stop()
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        dtype: str = DEFAULT_DTYPE,
    ) -> None:
        """
        Initialise the output engine.

        Args:
            device_index: PortAudio output device index.
            sample_rate:  Samples per second (Hz).
            channels:     Number of output channels.
            chunk_size:   Frames per callback.
            buffer_size:  Max frames held in the internal queue.
            dtype:        NumPy dtype for audio samples.
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.dtype = dtype

        self._stream: Optional[sd.OutputStream] = None
        # Keep enough buffer for ~170ms at 256-frame chunks @ 48kHz  (32 chunks)
        # to absorb any processing jitter without dropping chunks.
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(buffer_size // chunk_size, 32)
        )
        self._running = False
        self._output_level: float = 0.0
        self._silence = np.zeros((chunk_size, channels), dtype=dtype)

    # ── Stream control ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the OutputStream and start the background pump."""
        if self._running:
            logger.warning("AudioOutput already running.")
            return

        try:
            self._stream = sd.OutputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                dtype=self.dtype,
                callback=self._sd_callback,
                latency="low",
            )
            self._stream.start()
            self._running = True
            logger.info(
                "AudioOutput started – device=%s sr=%d ch=%d chunk=%d",
                self.device_index,
                self.sample_rate,
                self.channels,
                self.chunk_size,
            )
        except Exception as exc:
            logger.error("Failed to start AudioOutput: %s", exc)
            raise

    def stop(self) -> None:
        """Stop the output stream and flush the queue."""
        if not self._running:
            return
        self._running = False
        # Drain the queue to unblock any blocking put()
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning("Error closing output stream: %s", exc)
            self._stream = None
        logger.info("AudioOutput stopped.")

    def is_running(self) -> bool:
        """Return *True* if the output stream is active."""
        return self._running

    # ── Data feed ─────────────────────────────────────────────────────────────

    def write(self, audio_data: np.ndarray) -> None:
        """
        Enqueue a chunk of audio for playback.

        Args:
            audio_data: NumPy array shaped ``(frames, channels)`` or
                        ``(frames,)`` for mono.
        """
        if not self._running:
            return

        # Ensure correct shape
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, self.channels)

        try:
            self._audio_queue.put_nowait(audio_data.astype(self.dtype))
        except queue.Full:
            # Drop the chunk – better to skip than block the processing thread
            logger.debug("Output queue full – dropping chunk.")

    @property
    def output_level(self) -> float:
        """Current output RMS level (0.0 – 1.0)."""
        return self._output_level

    # ── Internal callback ─────────────────────────────────────────────────────

    def _sd_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Drain the queue into PortAudio's output buffer."""
        if status:
            logger.debug("Output status flags: %s", status)

        try:
            chunk = self._audio_queue.get_nowait()
            # Resize if needed (shouldn't happen in normal operation)
            if len(chunk) < frames:
                padded = np.zeros((frames, self.channels), dtype=self.dtype)
                padded[: len(chunk)] = chunk
                chunk = padded
            elif len(chunk) > frames:
                chunk = chunk[:frames]
            outdata[:] = chunk
            self._output_level = float(np.sqrt(np.mean(chunk ** 2)))
        except queue.Empty:
            outdata[:] = self._silence
            self._output_level = 0.0
