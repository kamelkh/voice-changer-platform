"""
Main application controller.

Owns all subsystems and wires them together:
  AudioDeviceManager → AudioStream → AudioPipeline
                     ↑                    ↑
               ProfileManager        RVCEngine / ModelManager
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.audio.devices import AudioDeviceManager
from src.audio.stream import AudioStream
from src.engine.model_manager import ModelManager
from src.engine.pipeline import AudioPipeline
from src.engine.rvc_engine import RVCEngine
from src.profiles.profile import VoiceProfile
from src.profiles.profile_manager import ProfileManager
from src.utils.constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CHANNELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLE_RATE,
    LATENCY_PRESETS,
    SETTINGS_FILE,
)
from src.utils.logger import get_logger, setup_root_logger

logger = get_logger(__name__)


class VoiceChangerApp:
    """
    Top-level application controller.

    Instantiate this class, then pass it to :class:`~src.ui.main_window.MainWindow`
    to run the full GUI application – or drive it headlessly for testing.

    Example::

        app = VoiceChangerApp()
        app.load_settings()
        app.start()          # begins audio streaming
        ...
        app.stop()
        app.shutdown()
    """

    def __init__(self) -> None:
        setup_root_logger()
        logger.info("Initialising VoiceChangerApp…")

        # Settings
        self._settings: dict = {}
        self.load_settings()

        # Subsystems
        self.device_manager = AudioDeviceManager()
        self.profile_manager = ProfileManager()
        self.model_manager = ModelManager()
        self.pipeline = AudioPipeline()

        self._rvc_engine: Optional[RVCEngine] = None

        self.stream: Optional[AudioStream] = None

        # Load profiles
        self.profile_manager.load_all()
        self.profile_manager.add_on_change_callback(self._on_profile_changed)

        # Restore last-used profile
        last_profile_key = self._settings.get("last_profile", "deep_male")
        if last_profile_key in self.profile_manager.get_profile_names():
            self.profile_manager.activate(last_profile_key)

        self._init_stream()
        logger.info("VoiceChangerApp ready.")

    # ── Settings ──────────────────────────────────────────────────────────────

    def load_settings(self) -> None:
        """Load settings from ``config/settings.json``."""
        if SETTINGS_FILE.exists():
            try:
                self._settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                logger.debug("Settings loaded from %s", SETTINGS_FILE)
            except Exception as exc:
                logger.warning("Failed to load settings: %s – using defaults.", exc)
                self._settings = {}
        else:
            logger.info("Settings file not found – using defaults.")
            self._settings = {}

    def save_settings(self) -> None:
        """Persist current settings to ``config/settings.json``."""
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(
                json.dumps(self._settings, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save settings: %s", exc)

    def _get(self, *keys, default=None):
        """Nested dict getter with a default."""
        val = self._settings
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    # ── Stream initialisation ─────────────────────────────────────────────────

    def _init_stream(self) -> None:
        """Create the :class:`~src.audio.stream.AudioStream` from settings."""
        audio_cfg = self._settings.get("audio", {})
        sample_rate = audio_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE)
        channels = audio_cfg.get("channels", DEFAULT_CHANNELS)
        chunk_size = audio_cfg.get("chunk_size", DEFAULT_CHUNK_SIZE)
        buffer_size = audio_cfg.get("buffer_size", DEFAULT_BUFFER_SIZE)

        # Resolve devices
        input_idx = self._resolve_device("input")
        output_idx = self._resolve_device("output", prefer_vbcable=True)

        self.stream = AudioStream(
            input_device=input_idx,
            output_device=output_idx,
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
        )
        self.stream.set_processor(self.pipeline.process)

    def _resolve_device(self, kind: str, prefer_vbcable: bool = False) -> Optional[int]:
        """Return a PortAudio device index for *kind* ('input'/'output')."""
        saved_name = self._settings.get(f"default_{kind}_device", "")
        if saved_name:
            dev = self.device_manager.find_device_by_name(saved_name)
            if dev:
                logger.debug("Using saved %s device: %s", kind, dev.name)
                return dev.index

        if prefer_vbcable:
            vb = self.device_manager.detect_vbcable_output()
            if vb:
                return vb.index

        if kind == "input":
            default = self.device_manager.get_default_input_device()
        else:
            default = self.device_manager.get_default_output_device()

        return default.index if default else None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start audio streaming.

        Returns:
            *True* if started successfully.
        """
        if self.stream is None:
            self._init_stream()

        try:
            self.stream.start()
            logger.info("Audio stream started.")
            return True
        except Exception as exc:
            logger.error("Failed to start audio stream: %s", exc)
            return False

    def stop(self) -> None:
        """Stop audio streaming."""
        if self.stream and self.stream.is_running:
            self.stream.stop()
            logger.info("Audio stream stopped.")

    def shutdown(self) -> None:
        """Cleanly shut down all subsystems."""
        self.stop()
        if self._rvc_engine and self._rvc_engine.is_loaded:
            self._rvc_engine.unload_model()
        self.save_settings()
        logger.info("VoiceChangerApp shutdown complete.")

    # ── Device management ─────────────────────────────────────────────────────

    def set_input_device(self, device_index: Optional[int]) -> None:
        """Switch the input device on the running stream."""
        if self.stream:
            self.stream.set_input_device(device_index)

    def set_output_device(self, device_index: Optional[int]) -> None:
        """Switch the output device on the running stream."""
        if self.stream:
            self.stream.set_output_device(device_index)

    # ── Profile management ────────────────────────────────────────────────────

    def apply_profile(self, profile: VoiceProfile) -> None:
        """Apply a profile to the pipeline (and load RVC model if needed)."""
        self.pipeline.load_from_profile(profile)

        if profile.use_ai and profile.ai_model_path:
            self._load_rvc_for_profile(profile)
        else:
            self.pipeline.set_rvc_engine(None)
            self._rvc_engine = None

    def _on_profile_changed(self, profile: VoiceProfile) -> None:
        self.apply_profile(profile)

    def _load_rvc_for_profile(self, profile: VoiceProfile) -> None:
        """Load/configure RVC engine from profile settings."""
        use_gpu = self._get("processing", "use_gpu", default=True)

        if self._rvc_engine is None:
            self._rvc_engine = RVCEngine(
                use_gpu=bool(use_gpu),
                f0_method=profile.ai_f0_method,
                pitch_shift=profile.ai_pitch_shift,
                index_rate=profile.ai_index_rate,
                protect=profile.ai_protect,
            )
        else:
            self._rvc_engine.set_params(
                pitch_shift=profile.ai_pitch_shift,
                index_rate=profile.ai_index_rate,
                protect=profile.ai_protect,
                f0_method=profile.ai_f0_method,
            )

        model_path = Path(profile.ai_model_path)
        if not model_path.is_absolute():
            model_path = Path(".") / model_path

        if model_path.exists():
            self._rvc_engine.load_model(model_path)
            self.pipeline.set_rvc_engine(self._rvc_engine)
        else:
            logger.warning(
                "AI model file not found: %s – AI conversion disabled.", model_path
            )
            self.pipeline.set_rvc_engine(None)

    # ── Latency preset ────────────────────────────────────────────────────────

    def apply_latency_preset(self, preset_name: str) -> None:
        """
        Switch to a named latency preset (e.g. 'ultra', 'low', 'medium', 'safe').

        Restarts the audio stream if currently running so the new buffer
        size takes effect immediately.
        """
        preset = LATENCY_PRESETS.get(preset_name)
        if preset is None:
            logger.warning("Unknown latency preset: %s – ignored.", preset_name)
            return

        chunk_size  = preset["chunk_size"]
        buffer_size = preset["buffer_size"]
        logger.info("Applying latency preset '%s': chunk=%d buffer=%d",
                    preset_name, chunk_size, buffer_size)

        # Update settings dict so they survive a stream restart
        self._settings.setdefault("audio", {})
        self._settings["audio"]["chunk_size"]  = chunk_size
        self._settings["audio"]["buffer_size"] = buffer_size
        self._settings["processing"]["latency_preset"] = preset_name

        was_running = bool(self.stream and self.stream.is_running)
        self.stop()
        self._init_stream()
        if was_running:
            self.start()

    # ── Effect parameter updates ──────────────────────────────────────────────

    def update_effect_param(self, param_name: str, value: float) -> None:
        """Update a single effect parameter in the running pipeline."""
        from src.engine.effects import (  # noqa: PLC0415
            FormantShifter,
            NoiseGate,
            PitchShifter,
            ReverbEffect,
            VolumeControl,
        )

        mapping = {
            "pitch_shift": (PitchShifter, "semitones"),
            "formant_shift": (FormantShifter, "semitones"),
            "reverb_level": (ReverbEffect, "wet_level"),
            "noise_gate": (NoiseGate, "threshold_db"),
            "gain": (VolumeControl, "gain"),
        }

        if param_name not in mapping:
            return

        effect_type, key = mapping[param_name]
        effect = self.pipeline.get_effect_by_type(effect_type)
        if effect is not None:
            effect.set_params({key: value})
            logger.debug("Effect param updated: %s.%s = %s", effect_type.__name__, key, value)
