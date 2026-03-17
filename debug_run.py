"""Debug launcher – runs the platform with full DEBUG logging in console."""
import logging
import sys
from pathlib import Path

# Ensure project root on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ── Force DEBUG on every console handler ──────────────────────────────────
import src.utils.logger as logmod

_orig_configure = logmod._configure_logger

def _debug_configure(logger: logging.Logger) -> None:
    """Wrap the original configure to force DEBUG on console."""
    _orig_configure(logger)
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.DEBUG)

logmod._configure_logger = _debug_configure

# ── Launch ────────────────────────────────────────────────────────────────
from src.main import main
sys.exit(main())
