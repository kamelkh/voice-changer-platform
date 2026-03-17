"""
Audio capture module – reads audio from a physical microphone.

Uses *sounddevice* for non-blocking, callback-based streaming.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

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

# Type alias for audio callback
AudioCallback = Callable[[np.ndarray], None]


class AudioCapture:
    """
    Capture audio from a physical microphone in real-time.

    The capture runs in a background thread via sounddevice's
    InputStream.  Callers register a callback that receives each
    chunk of audio as a ``numpy.ndarray`` with shape
    ``(chunk_size, channels)`` and dtype *float32*.

    Example::

        def on_audio(chunk: np.ndarray) -> None:
            process(chunk)

        capture = AudioCapture(device_index=0)
        capture.add_callback(on_audio)
        capture.start()
        ...
        capture.stop()
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
        Initialise the capture engine.

        Args:
            device_index:   PortAudio input device index.  *None* uses
                            the system default.
            sample_rate:    Samples per second (Hz).
            channels:       Number of input channels (1 = mono).
            chunk_size:     Frames per callback invocation.
            buffer_size:    Internal ring-buffer size in frames.
            dtype:          NumPy dtype for audio samples.
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.dtype = dtype

        self._stream: Optional[sd.InputStream] = None
        self._callbacks: list[AudioCallback] = []
        self._lock = threading.Lock()
        self._running = False

        # Ring buffer for monitoring / level metering
        self._ring_buffer: deque[np.ndarray] = deque(
            maxlen=buffer_size // chunk_size
        )

        # Volume level (RMS, updated each chunk)
        self._input_level: float = 0.0

    # ── Callback management ───────────────────────────────────────────────────

    def add_callback(self, cb: AudioCallback) -> None:
        """Register a function to be called with each audio chunk."""
        with self._lock:
            self._callbacks.append(cb)

    def remove_callback(self, cb: AudioCallback) -> None:
        """Unregister a previously added callback."""
        with self._lock:
            self._callbacks = [c for c in self._callbacks if c is not cb]

    # ── Stream control ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the InputStream and begin capturing audio."""
        if self._running:
            logger.warning("AudioCapture already running.")
            return

        logger.debug("[CAPTURE] opening InputStream: device=%s sr=%s ch=%s chunk=%s dtype=%s",
                     self.device_index, self.sample_rate, self.channels,
                     self.chunk_size, self.dtype)
        try:
            self._stream = sd.InputStream(
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
                "AudioCapture started – device=%s sr=%d ch=%d chunk=%d",
                self.device_index,
                self.sample_rate,
                self.channels,
                self.chunk_size,
            )
        except Exception as exc:
            logger.error("Failed to start AudioCapture: %s", exc)
            raise

    def stop(self) -> None:
        """Stop capturing and close the InputStream."""
        if not self._running:
            return
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning("Error closing capture stream: %s", exc)
            self._stream = None
        logger.info("AudioCapture stopped.")

    def is_running(self) -> bool:
        """Return *True* if the stream is active."""
        return self._running

    # ── Levels ────────────────────────────────────────────────────────────────

    @property
    def input_level(self) -> float:
        """Current input RMS level (0.0 – 1.0)."""
        return self._input_level

    # ── Internal callback ─────────────────────────────────────────────────────

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice on every audio chunk (background thread)."""
        if status:
            logger.debug("Capture status flags: %s", status)

        chunk = indata.copy()

        # Update RMS level
        self._input_level = float(np.sqrt(np.mean(chunk ** 2)))

        # Push to ring buffer
        self._ring_buffer.append(chunk)

        # Dispatch to registered callbacks
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(chunk)
            except Exception as exc:
                logger.error("Error in audio capture callback: %s", exc)
