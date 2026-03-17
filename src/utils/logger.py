"""
Logging configuration for the Voice Changer Platform.
"""
import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        _configure_logger(logger)
    return logger


def _configure_logger(logger: logging.Logger) -> None:
    """Configure handlers and formatters for a logger."""
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG and above)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"voice_changer_{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False


def setup_root_logger(level: int = logging.INFO) -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Logging level for the root logger.
    """
    root_logger = logging.getLogger("voice_changer")
    root_logger.setLevel(level)
    if not root_logger.handlers:
        _configure_logger(root_logger)
