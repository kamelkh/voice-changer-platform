"""
Entry point for the Voice Changer Platform.

Run with:
    python -m src.main
or:
    scripts\\run.bat
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure the project root (parent of this file's directory) is on sys.path so
# that "from src.xxx import ..." works whether launched via:
#   - python src/main.py
#   - python -m src.main
#   - the packaged VoiceChanger.exe  (PyInstaller sets sys._MEIPASS)
_here = Path(__file__).resolve().parent          # …/src
_root = _here.parent                              # …/voice-changer-platform
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# When frozen by PyInstaller, also add the _MEIPASS temp dir
if getattr(sys, "frozen", False):
    _mei = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    if str(_mei) not in sys.path:
        sys.path.insert(0, str(_mei))
    # Change cwd to _MEIPASS so relative config/model paths resolve
    os.chdir(_mei)


def main() -> int:
    """
    Launch the Voice Changer Platform desktop application.

    Returns:
        Exit code (0 = success).
    """
    try:
        from src.app import VoiceChangerApp  # noqa: PLC0415
        from src.ui.main_window import MainWindow  # noqa: PLC0415
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}")
        print("Run:  pip install -r requirements.txt")
        return 1

    try:
        app = VoiceChangerApp()
        window = MainWindow(app)
        window.run()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        from src.utils.logger import get_logger  # noqa: PLC0415
        logger = get_logger(__name__)
        logger.exception("Unhandled exception in main: %s", exc)
        # Show a visible error dialog so the window doesn't just vanish
        try:
            import tkinter as tk  # noqa: PLC0415
            from tkinter import messagebox  # noqa: PLC0415
            _root = tk.Tk()
            _root.withdraw()
            messagebox.showerror(
                "Voice Changer – Unexpected Error",
                f"The application crashed:\n\n{type(exc).__name__}: {exc}\n\n"
                f"Full details are in the logs/ folder.\n\n"
                f"{traceback.format_exc()[-800:]}",
            )
            _root.destroy()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
