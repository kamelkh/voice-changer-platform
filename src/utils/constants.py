"""
Application-wide constants for the Voice Changer Platform.
"""
import sys
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
# When frozen (PyInstaller onefile) all files land in sys._MEIPASS.
# When running from source, ROOT_DIR is the repo root (parent of src/).
if getattr(sys, "frozen", False):
    ROOT_DIR: Path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent

CONFIG_DIR: Path = ROOT_DIR / "config"
PROFILES_DIR: Path = CONFIG_DIR / "profiles"
MODELS_DIR: Path = ROOT_DIR / "models"
LOGS_DIR: Path = ROOT_DIR / "logs"

# ── Configuration files ───────────────────────────────────────────────────────
SETTINGS_FILE: Path = CONFIG_DIR / "settings.json"

# ── Audio defaults ────────────────────────────────────────────────────────────
# 48 kHz is native for WASAPI / most voice apps (Zoom, Discord, WhatsApp)
DEFAULT_SAMPLE_RATE: int = 48000
DEFAULT_CHANNELS: int = 1
# 1024 frames @ 16 kHz = 64 ms per chunk — good balance between latency and
# processing throughput (avoids per-chunk overhead dominating at 256 frames).
DEFAULT_CHUNK_SIZE: int = 1024
DEFAULT_BUFFER_SIZE: int = 4096
DEFAULT_DTYPE: str = "float32"

# ── Low-latency presets ───────────────────────────────────────────────────────
LATENCY_PRESETS: dict = {
    "ultra": {"chunk_size": 128,  "buffer_size": 512,  "label": "Ultra  (~3 ms)"},
    "low":   {"chunk_size": 256,  "buffer_size": 1024, "label": "Low    (~6 ms)"},
    "medium":{"chunk_size": 512,  "buffer_size": 2048, "label": "Medium (~11 ms)"},
    "safe":  {"chunk_size": 1024, "buffer_size": 4096, "label": "Safe   (~23 ms)"},
}

# ── VB-Audio Virtual Cable ────────────────────────────────────────────────────
VBCABLE_INPUT_NAME: str = "CABLE Input"   # Virtual mic input (we write here)
VBCABLE_OUTPUT_NAME: str = "CABLE Output" # Virtual mic output (apps read from here)
VBCABLE_ALT_NAMES: list[str] = ["VB-Audio", "Virtual Cable", "VB-Cable"]

# ── Processing limits ─────────────────────────────────────────────────────────
MAX_LATENCY_MS: int = 50
MIN_PITCH_SHIFT: float = -12.0   # semitones
MAX_PITCH_SHIFT: float = 12.0    # semitones
MIN_FORMANT_SHIFT: float = -6.0  # semitones
MAX_FORMANT_SHIFT: float = 6.0   # semitones
MIN_REVERB: float = 0.0
MAX_REVERB: float = 1.0
MIN_NOISE_GATE_DB: float = -80.0
MAX_NOISE_GATE_DB: float = 0.0
MIN_GAIN: float = 0.0
MAX_GAIN: float = 4.0

# ── RVC defaults ──────────────────────────────────────────────────────────────
RVC_DEFAULT_F0_METHOD: str = "rmvpe"
RVC_DEFAULT_INDEX_RATE: float = 0.5
RVC_DEFAULT_PROTECT: float = 0.33
RVC_DEFAULT_FILTER_RADIUS: int = 3
RVC_MODEL_EXTENSIONS: list[str] = [".pth"]
RVC_INDEX_EXTENSIONS: list[str] = [".index"]

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE: str = "AI Voice Changer"
APP_VERSION: str = "2.0.0"
DEFAULT_WINDOW_WIDTH: int = 960
DEFAULT_WINDOW_HEIGHT: int = 700

# Latency colour thresholds
LATENCY_GOOD_MS: float = 15.0    # green
LATENCY_WARN_MS: float = 35.0    # yellow
LATENCY_BAD_MS: float  = 60.0    # red
LATENCY_GOOD_COLOR: str = "#22c55e"
LATENCY_WARN_COLOR: str = "#eab308"
LATENCY_BAD_COLOR:  str = "#ef4444"
DARK_BG: str = "#1e1e2e"
DARK_SURFACE: str = "#2a2a3e"
DARK_ACCENT: str = "#7c3aed"
DARK_ACCENT_HOVER: str = "#6d28d9"
ACTIVE_COLOR: str = "#22c55e"
INACTIVE_COLOR: str = "#ef4444"
TEXT_COLOR: str = "#e2e8f0"
SUBTEXT_COLOR: str = "#94a3b8"
CARD_BG: str = "#313145"
CARD_SELECTED_BG: str = "#4c1d95"
