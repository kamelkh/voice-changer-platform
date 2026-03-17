"""
Audio device enumeration and selection utilities.

Provides helpers to list microphones and speakers/virtual cables,
auto-detect VB-Audio Virtual Cable, and look up device indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sounddevice as sd

from src.utils.constants import VBCABLE_ALT_NAMES, VBCABLE_INPUT_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AudioDevice:
    """Represents a single audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    host_api: int

    @property
    def is_input(self) -> bool:
        """True when the device can capture audio."""
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        """True when the device can play audio."""
        return self.max_output_channels > 0

    def __str__(self) -> str:  # noqa: D105
        return f"[{self.index}] {self.name}"


class AudioDeviceManager:
    """
    Manage audio device listing and selection.

    Uses the *sounddevice* library which wraps PortAudio and works
    on Windows, macOS, and Linux.
    """

    def __init__(self) -> None:
        self._devices: list[AudioDevice] = []
        self.refresh()

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-query PortAudio for the current device list."""
        self._devices = []
        try:
            raw = sd.query_devices()
            for idx, dev in enumerate(raw):
                self._devices.append(
                    AudioDevice(
                        index=idx,
                        name=dev["name"],
                        max_input_channels=dev["max_input_channels"],
                        max_output_channels=dev["max_output_channels"],
                        default_sample_rate=dev["default_samplerate"],
                        host_api=dev["hostapi"],
                    )
                )
            logger.debug("Found %d audio devices.", len(self._devices))
        except Exception as exc:
            logger.error("Failed to query audio devices: %s", exc)

    def get_input_devices(self) -> list[AudioDevice]:
        """Return all devices that have at least one input channel."""
        return [d for d in self._devices if d.is_input]

    def get_output_devices(self) -> list[AudioDevice]:
        """Return all devices that have at least one output channel."""
        return [d for d in self._devices if d.is_output]

    def get_all_devices(self) -> list[AudioDevice]:
        """Return the full device list."""
        return list(self._devices)

    def find_device_by_name(self, name: str, partial: bool = True) -> Optional[AudioDevice]:
        """
        Look up a device by name (case-insensitive).

        Args:
            name: Device name to search for.
            partial: When *True* accept substring matches.

        Returns:
            First matching :class:`AudioDevice`, or *None*.
        """
        name_lower = name.lower()
        for dev in self._devices:
            dev_lower = dev.name.lower()
            if partial and name_lower in dev_lower:
                return dev
            if not partial and name_lower == dev_lower:
                return dev
        return None

    def find_device_by_index(self, index: int) -> Optional[AudioDevice]:
        """Return the device with the given PortAudio index, or *None*."""
        for dev in self._devices:
            if dev.index == index:
                return dev
        return None

    def detect_vbcable_output(self) -> Optional[AudioDevice]:
        """
        Auto-detect the VB-Audio Virtual Cable output device.

        The *output* here means the PortAudio output device that maps to
        the CABLE Input (i.e., the virtual microphone's write end).

        Returns:
            Detected :class:`AudioDevice` or *None* if not installed.
        """
        # Try primary name first
        device = self.find_device_by_name(VBCABLE_INPUT_NAME)
        if device:
            logger.info("VB-Cable detected: %s", device.name)
            return device

        # Fallback to alternative name patterns
        for alt_name in VBCABLE_ALT_NAMES:
            device = self.find_device_by_name(alt_name)
            if device and device.is_output:
                logger.info("VB-Cable detected (alt name): %s", device.name)
                return device

        logger.warning(
            "VB-Audio Virtual Cable not found. "
            "Install it from https://vb-audio.com/Cable/"
        )
        return None

    def get_default_input_device(self) -> Optional[AudioDevice]:
        """Return the system default input device."""
        try:
            idx = sd.default.device[0]
            if idx is not None and idx >= 0:
                return self.find_device_by_index(int(idx))
        except Exception as exc:
            logger.debug("Could not get default input device: %s", exc)
        return None

    def get_default_output_device(self) -> Optional[AudioDevice]:
        """Return the system default output device."""
        try:
            idx = sd.default.device[1]
            if idx is not None and idx >= 0:
                return self.find_device_by_index(int(idx))
        except Exception as exc:
            logger.debug("Could not get default output device: %s", exc)
        return None

    def print_devices(self) -> None:
        """Pretty-print all devices to stdout."""
        print("\n=== Input Devices ===")
        for dev in self.get_input_devices():
            print(f"  {dev}")
        print("\n=== Output Devices ===")
        for dev in self.get_output_devices():
            print(f"  {dev}")
        print()
