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

    # ── Host-API helpers ──────────────────────────────────────────────────────

    def _wasapi_api_index(self) -> int | None:
        """Return the PortAudio host-API index for WASAPI, or None."""
        try:
            for i, api in enumerate(sd.query_hostapis()):
                if "wasapi" in api["name"].lower():
                    return i
        except Exception:
            pass
        return None

    def _best_devices(self) -> list[AudioDevice]:
        """Return WASAPI-only devices when available, else all devices."""
        wasapi_idx = self._wasapi_api_index()
        if wasapi_idx is not None:
            wasapi = [d for d in self._devices if d.host_api == wasapi_idx]
            if wasapi:
                return wasapi
        return self._devices

    # ── Public API ────────────────────────────────────────────────────────────

    def get_input_devices(self) -> list[AudioDevice]:
        """
        Return microphone / capture devices.

        Uses WASAPI on Windows so each device appears only once and
        pure playback devices are excluded from the list.
        """
        devs = self._best_devices()
        # Prefer pure-input devices (no output channels), but keep
        # full-duplex ones too (e.g. Bluetooth headsets).
        pure   = [d for d in devs if d.max_input_channels > 0 and d.max_output_channels == 0]
        duplex = [d for d in devs if d.max_input_channels > 0 and d.max_output_channels > 0]
        return pure + duplex

    def get_output_devices(self) -> list[AudioDevice]:
        """
        Return speaker / virtual cable playback devices.

        Uses WASAPI on Windows so microphones do *not* appear here.
        VB-Audio CABLE Input is always included when present.
        """
        devs = self._best_devices()
        # Pure output only (no input channels) – this is what we want
        # to show in the Output dropdown (speakers, VB-Cable, etc.).
        pure   = [d for d in devs if d.max_output_channels > 0 and d.max_input_channels == 0]
        # Also include full-duplex devices whose name hints at a virtual cable
        # or audio interface (we still want CABLE Input here).
        cable_names = (VBCABLE_INPUT_NAME.lower(),) + tuple(n.lower() for n in VBCABLE_ALT_NAMES)
        duplex_output = [
            d for d in devs
            if d.max_output_channels > 0 and d.max_input_channels > 0
            and any(n in d.name.lower() for n in cable_names)
        ]
        seen = {d.index for d in pure}
        for d in duplex_output:
            if d.index not in seen:
                pure.append(d)
        return pure

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

        Searches the full device list (all host APIs) so CABLE Input is
        found even when WASAPI filtering is active.
        """
        # Search ALL devices (not just WASAPI-filtered) for reliability
        for dev in self._devices:
            if VBCABLE_INPUT_NAME.lower() in dev.name.lower() and dev.is_output:
                logger.info("VB-Cable detected: %s", dev.name)
                return dev

        for alt_name in VBCABLE_ALT_NAMES:
            for dev in self._devices:
                if alt_name.lower() in dev.name.lower() and dev.is_output:
                    logger.info("VB-Cable detected (alt name): %s", dev.name)
                    return dev

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
