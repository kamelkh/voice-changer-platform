# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the AI Voice Changer Platform.

Build:
    pyinstaller voice_changer.spec --clean

Output: dist/VoiceChanger.exe  (single-file, no console)
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH)   # noqa: F821  (SPECPATH is injected by PyInstaller)

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [str(ROOT / "src" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # Ship the config directory (settings.json + profile JSON files)
        (str(ROOT / "config"),  "config"),
        # Ship the models directory stub (user places .pth files here)
        (str(ROOT / "models"),  "models"),
        # Ship the assets directory if it exists
        # (str(ROOT / "assets"), "assets"),
    ],
    hiddenimports=[
        # sounddevice requires PortAudio DLL lookup at runtime
        "sounddevice",
        "soundfile",
        # scipy sub-modules used by effects.py
        "scipy.signal",
        "scipy.signal.signaltools",
        # NumPy
        "numpy",
        "numpy.core._multiarray_umath",
        # TKinter themes
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
        # Logging
        "logging.handlers",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy packages not needed in the packaged app
        "matplotlib",
        "IPython",
        "jupyter",
        "pytest",
        "sphinx",
        "sklearn",
        "scikit_learn",
        "django",
        "sentry_sdk",
        "pyarrow",
        "datasets",
        "pandas",
        "sqlalchemy",
        "torchvision",
        # torch & torchaudio are large — exclude from analysis;
        # they will be loaded at runtime if the user has them installed.
        # Remove these two lines if you want AI inference bundled:
        "torch",
        "torchaudio",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

# ── PYZ (bytecode archive) ────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data)   # noqa: F821

# ── EXE ───────────────────────────────────────────────────────────────────────
exe = EXE(            # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VoiceChanger",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,          # enable UPX compression (install UPX separately)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,     # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="assets/icon.ico",   # uncomment once you have an icon
    version_file=None,
    # Set Windows metadata
    uac_admin=False,
)
