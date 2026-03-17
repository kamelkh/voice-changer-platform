"""
Unit tests for audio device discovery.

These tests run without real audio hardware by mocking sounddevice.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_device(
    name: str,
    max_in: int = 2,
    max_out: int = 2,
    sr: float = 44100.0,
    host_api: int = 0,
) -> dict:
    return {
        "name": name,
        "max_input_channels": max_in,
        "max_output_channels": max_out,
        "default_samplerate": sr,
        "hostapi": host_api,
    }


_FAKE_DEVICES = [
    _fake_device("Microphone (Realtek HD Audio)", max_out=0),
    _fake_device("Speakers (Realtek HD Audio)", max_in=0),
    _fake_device("CABLE Input (VB-Audio Virtual Cable)", max_in=0),
    _fake_device("CABLE Output (VB-Audio Virtual Cable)", max_out=0),
    _fake_device("Stereo Mix", max_in=2, max_out=2),
]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAudioDeviceManager(unittest.TestCase):
    """Tests for :class:`~src.audio.devices.AudioDeviceManager`."""

    def _make_manager(self) -> "AudioDeviceManager":  # noqa: F821
        with patch("sounddevice.query_devices", return_value=_FAKE_DEVICES), \
             patch("sounddevice.default", new=MagicMock(device=(0, 1))):
            from src.audio.devices import AudioDeviceManager
            return AudioDeviceManager()

    def test_input_devices_only_have_input_channels(self) -> None:
        mgr = self._make_manager()
        for dev in mgr.get_input_devices():
            self.assertGreater(dev.max_input_channels, 0)

    def test_output_devices_only_have_output_channels(self) -> None:
        mgr = self._make_manager()
        for dev in mgr.get_output_devices():
            self.assertGreater(dev.max_output_channels, 0)

    def test_total_device_count(self) -> None:
        mgr = self._make_manager()
        self.assertEqual(len(mgr.get_all_devices()), len(_FAKE_DEVICES))

    def test_find_device_by_name_partial(self) -> None:
        mgr = self._make_manager()
        dev = mgr.find_device_by_name("CABLE Input", partial=True)
        self.assertIsNotNone(dev)
        self.assertIn("CABLE Input", dev.name)

    def test_find_device_by_name_exact_not_found(self) -> None:
        mgr = self._make_manager()
        dev = mgr.find_device_by_name("nonexistent device", partial=False)
        self.assertIsNone(dev)

    def test_detect_vbcable_output(self) -> None:
        mgr = self._make_manager()
        vb = mgr.detect_vbcable_output()
        self.assertIsNotNone(vb)
        self.assertIn("CABLE", vb.name)

    def test_find_device_by_index(self) -> None:
        mgr = self._make_manager()
        dev = mgr.find_device_by_index(2)
        self.assertIsNotNone(dev)
        self.assertEqual(dev.index, 2)

    def test_find_nonexistent_index_returns_none(self) -> None:
        mgr = self._make_manager()
        dev = mgr.find_device_by_index(999)
        self.assertIsNone(dev)

    def test_device_str(self) -> None:
        mgr = self._make_manager()
        dev = mgr.get_all_devices()[0]
        self.assertIn(str(dev.index), str(dev))
        self.assertIn(dev.name, str(dev))


if __name__ == "__main__":
    unittest.main()
