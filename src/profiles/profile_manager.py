"""
Profile manager – load, save, switch, import, and export voice profiles.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

from src.profiles.profile import VoiceProfile
from src.utils.constants import PROFILES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for profile-change listeners
ProfileChangeCallback = Callable[[VoiceProfile], None]


class ProfileManager:
    """
    Manage the collection of :class:`~src.profiles.profile.VoiceProfile` objects.

    Profiles are persisted as JSON files in ``config/profiles/``.
    The active profile is applied to the processing pipeline in real-time.

    Example::

        pm = ProfileManager()
        pm.load_all()
        pm.activate("female")
        pm.add_on_change_callback(lambda p: pipeline.load_from_profile(p))
    """

    def __init__(self, profiles_dir: Path = PROFILES_DIR) -> None:
        self.profiles_dir = profiles_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self._profiles: dict[str, VoiceProfile] = {}
        self._active_profile: Optional[VoiceProfile] = None
        self._change_callbacks: list[ProfileChangeCallback] = []

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Load all JSON profiles from the profiles directory."""
        self._profiles.clear()
        for json_file in sorted(self.profiles_dir.glob("*.json")):
            try:
                profile = VoiceProfile.load(json_file)
                key = json_file.stem
                self._profiles[key] = profile
                logger.debug("Loaded profile: %s (%s)", key, profile.name)
            except Exception as exc:
                logger.warning("Failed to load profile '%s': %s", json_file.name, exc)

        logger.info("Loaded %d profile(s).", len(self._profiles))

    def reload(self) -> None:
        """Reload all profiles from disk (non-destructive refresh)."""
        self.load_all()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def get_profile(self, key: str) -> Optional[VoiceProfile]:
        """Return a profile by its file-stem key, or *None*."""
        return self._profiles.get(key)

    def get_all_profiles(self) -> list[tuple[str, VoiceProfile]]:
        """Return (key, profile) pairs in the order they were loaded."""
        return list(self._profiles.items())

    def get_profile_names(self) -> list[str]:
        """Return all profile keys."""
        return list(self._profiles.keys())

    def add_profile(self, key: str, profile: VoiceProfile, save: bool = True) -> None:
        """
        Add or replace a profile.

        Args:
            key:     Unique identifier (used as the JSON filename stem).
            profile: The profile to store.
            save:    When *True*, write to disk immediately.
        """
        self._profiles[key] = profile
        if save:
            self.save_profile(key)
        logger.info("Profile added: %s (%s)", key, profile.name)

    def save_profile(self, key: str) -> bool:
        """
        Persist a profile to ``config/profiles/<key>.json``.

        Returns:
            *True* on success.
        """
        profile = self._profiles.get(key)
        if profile is None:
            logger.warning("Cannot save unknown profile key: %s", key)
            return False

        path = self.profiles_dir / f"{key}.json"
        try:
            profile.save(path)
            logger.debug("Saved profile '%s' → %s", key, path)
            return True
        except Exception as exc:
            logger.error("Failed to save profile '%s': %s", key, exc)
            return False

    def delete_profile(self, key: str) -> bool:
        """
        Remove a profile from memory and delete its JSON file.

        Returns:
            *True* if deleted.
        """
        if key not in self._profiles:
            logger.warning("Profile not found: %s", key)
            return False

        if self._active_profile is self._profiles[key]:
            self._active_profile = None

        path = self.profiles_dir / f"{key}.json"
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not delete profile file: %s", exc)

        del self._profiles[key]
        logger.info("Deleted profile: %s", key)
        return True

    # ── Activation ────────────────────────────────────────────────────────────

    def activate(self, key: str) -> bool:
        """
        Set the active profile and fire callbacks.

        Args:
            key: Profile key to activate.

        Returns:
            *True* if the profile was found and activated.
        """
        profile = self._profiles.get(key)
        if profile is None:
            logger.warning("Unknown profile key: %s", key)
            return False

        self._active_profile = profile
        logger.info("Active profile: %s – %s", key, profile.name)
        self._fire_callbacks(profile)
        return True

    @property
    def active_profile(self) -> Optional[VoiceProfile]:
        """The currently active :class:`VoiceProfile`, or *None*."""
        return self._active_profile

    @property
    def active_key(self) -> Optional[str]:
        """Key of the active profile, or *None*."""
        if self._active_profile is None:
            return None
        for key, profile in self._profiles.items():
            if profile is self._active_profile:
                return key
        return None

    # ── Import / Export ───────────────────────────────────────────────────────

    def import_profile(self, source_path: str | Path) -> Optional[str]:
        """
        Import a profile JSON file into the profiles directory.

        Args:
            source_path: Path to the .json file to import.

        Returns:
            The new profile key on success, *None* on failure.
        """
        source = Path(source_path)
        if not source.exists():
            logger.error("Import source not found: %s", source)
            return None

        try:
            profile = VoiceProfile.load(source)
            key = source.stem
            dest = self.profiles_dir / source.name
            shutil.copy2(source, dest)
            self._profiles[key] = profile
            logger.info("Imported profile: %s", key)
            return key
        except Exception as exc:
            logger.error("Failed to import profile: %s", exc)
            return None

    def export_profile(self, key: str, destination: str | Path) -> bool:
        """
        Copy a profile JSON to a user-specified path.

        Args:
            key:         Profile key.
            destination: Target file path (should end in .json).

        Returns:
            *True* on success.
        """
        profile = self._profiles.get(key)
        if profile is None:
            logger.error("Cannot export unknown profile: %s", key)
            return False

        dest = Path(destination)
        try:
            profile.save(dest)
            logger.info("Exported profile '%s' → %s", key, dest)
            return True
        except Exception as exc:
            logger.error("Export failed: %s", exc)
            return False

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def add_on_change_callback(self, callback: ProfileChangeCallback) -> None:
        """Register a listener that fires when the active profile changes."""
        self._change_callbacks.append(callback)

    def remove_on_change_callback(self, callback: ProfileChangeCallback) -> None:
        """Unregister a profile-change listener."""
        self._change_callbacks = [c for c in self._change_callbacks if c is not callback]

    def _fire_callbacks(self, profile: VoiceProfile) -> None:
        for cb in self._change_callbacks:
            try:
                cb(profile)
            except Exception as exc:
                logger.error("Profile change callback error: %s", exc)
