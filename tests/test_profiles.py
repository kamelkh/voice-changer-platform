"""
Unit tests for VoiceProfile and ProfileManager.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestVoiceProfile(unittest.TestCase):
    """Tests for :class:`~src.profiles.profile.VoiceProfile`."""

    def _make_profile(self, **kwargs):
        from src.profiles.profile import VoiceProfile
        return VoiceProfile(name="Test Voice", description="A test profile", **kwargs)

    def test_default_values(self) -> None:
        from src.profiles.profile import VoiceProfile
        p = VoiceProfile()
        self.assertEqual(p.pitch_shift, 0.0)
        self.assertEqual(p.reverb_level, 0.0)
        self.assertFalse(p.use_ai)

    def test_to_dict_roundtrip(self) -> None:
        from src.profiles.profile import VoiceProfile
        p = self._make_profile(pitch_shift=-4.0, reverb_level=0.1)
        d = p.to_dict()
        restored = VoiceProfile.from_dict(d)
        self.assertEqual(restored.pitch_shift, p.pitch_shift)
        self.assertEqual(restored.reverb_level, p.reverb_level)
        self.assertEqual(restored.name, p.name)

    def test_json_roundtrip(self) -> None:
        from src.profiles.profile import VoiceProfile
        p = self._make_profile(pitch_shift=5.0, gain=1.5)
        restored = VoiceProfile.from_json(p.to_json())
        self.assertEqual(restored.pitch_shift, 5.0)
        self.assertEqual(restored.gain, 1.5)

    def test_save_and_load(self) -> None:
        from src.profiles.profile import VoiceProfile
        p = self._make_profile(pitch_shift=3.0, formant_shift=2.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test_profile.json"
            p.save(path)
            self.assertTrue(path.exists())
            loaded = VoiceProfile.load(path)
            self.assertEqual(loaded.pitch_shift, 3.0)
            self.assertEqual(loaded.formant_shift, 2.0)

    def test_from_dict_ignores_unknown_keys(self) -> None:
        from src.profiles.profile import VoiceProfile
        data = {
            "name": "Safe",
            "pitch_shift": 1.0,
            "unknown_future_field": "value",
        }
        # Should not raise
        p = VoiceProfile.from_dict(data)
        self.assertEqual(p.name, "Safe")

    def test_copy_is_independent(self) -> None:
        p = self._make_profile(pitch_shift=2.0)
        clone = p.copy()
        clone.pitch_shift = 9.0
        self.assertEqual(p.pitch_shift, 2.0)

    def test_apply_overrides(self) -> None:
        p = self._make_profile(pitch_shift=2.0, gain=1.0)
        modified = p.apply_overrides(pitch_shift=5.0)
        self.assertEqual(modified.pitch_shift, 5.0)
        self.assertEqual(modified.gain, 1.0)  # unchanged

    def test_str_includes_name(self) -> None:
        p = self._make_profile()
        self.assertIn("Test Voice", str(p))

    def test_str_includes_ai_indicator(self) -> None:
        p = self._make_profile(use_ai=True)
        self.assertIn("[AI]", str(p))


class TestProfileManager(unittest.TestCase):
    """Tests for :class:`~src.profiles.profile_manager.ProfileManager`."""

    def _make_manager(self, tmp_dir: str):
        from src.profiles.profile_manager import ProfileManager
        return ProfileManager(profiles_dir=Path(tmp_dir))

    def _write_profile(self, directory: Path, key: str, data: dict) -> None:
        path = directory / f"{key}.json"
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_load_all_discovers_profiles(self) -> None:
        from src.profiles.profile import VoiceProfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdir = Path(tmp_dir)
            self._write_profile(pdir, "deep_male", {"name": "Deep Male", "pitch_shift": -4.0})
            self._write_profile(pdir, "female", {"name": "Female", "pitch_shift": 5.0})

            pm = self._make_manager(tmp_dir)
            pm.load_all()

            names = pm.get_profile_names()
            self.assertIn("deep_male", names)
            self.assertIn("female", names)

    def test_activate_fires_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdir = Path(tmp_dir)
            self._write_profile(pdir, "robot", {"name": "Robot"})

            pm = self._make_manager(tmp_dir)
            pm.load_all()

            received = []
            pm.add_on_change_callback(lambda p: received.append(p.name))
            pm.activate("robot")

            self.assertEqual(len(received), 1)
            self.assertEqual(received[0], "Robot")

    def test_active_profile_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdir = Path(tmp_dir)
            self._write_profile(pdir, "robot", {"name": "Robot"})

            pm = self._make_manager(tmp_dir)
            pm.load_all()
            pm.activate("robot")

            self.assertIsNotNone(pm.active_profile)
            self.assertEqual(pm.active_profile.name, "Robot")
            self.assertEqual(pm.active_key, "robot")

    def test_activate_unknown_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pm = self._make_manager(tmp_dir)
            pm.load_all()
            result = pm.activate("nonexistent")
            self.assertFalse(result)

    def test_save_and_reload_profile(self) -> None:
        from src.profiles.profile import VoiceProfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            pm = self._make_manager(tmp_dir)
            profile = VoiceProfile(name="Custom", pitch_shift=7.0)
            pm.add_profile("custom", profile)

            pm2 = self._make_manager(tmp_dir)
            pm2.load_all()
            loaded = pm2.get_profile("custom")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.pitch_shift, 7.0)

    def test_delete_profile(self) -> None:
        from src.profiles.profile import VoiceProfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            pm = self._make_manager(tmp_dir)
            pm.add_profile("to_delete", VoiceProfile(name="Delete Me"))
            pm.delete_profile("to_delete")
            self.assertNotIn("to_delete", pm.get_profile_names())

    def test_import_export_profile(self) -> None:
        from src.profiles.profile import VoiceProfile
        with tempfile.TemporaryDirectory() as src_dir, \
             tempfile.TemporaryDirectory() as dest_dir:

            pm_src = self._make_manager(src_dir)
            pm_src.add_profile("export_test", VoiceProfile(name="Exported", gain=1.8))

            export_path = Path(dest_dir) / "export_test.json"
            ok = pm_src.export_profile("export_test", export_path)
            self.assertTrue(ok)

            pm_dest = self._make_manager(dest_dir)
            pm_dest.load_all()
            loaded = pm_dest.get_profile("export_test")
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded.gain, 1.8)


if __name__ == "__main__":
    unittest.main()
