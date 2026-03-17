"""
Status bar widget – shows latency, input/output volume meters.
"""
from __future__ import annotations

import tkinter as tk
from typing import Optional

from src.utils.constants import (
    ACTIVE_COLOR,
    DARK_BG,
    DARK_SURFACE,
    INACTIVE_COLOR,
    LATENCY_BAD_COLOR,
    LATENCY_BAD_MS,
    LATENCY_GOOD_COLOR,
    LATENCY_GOOD_MS,
    LATENCY_WARN_COLOR,
    LATENCY_WARN_MS,
    SUBTEXT_COLOR,
    TEXT_COLOR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _latency_color(ms: float) -> str:
    """Return a hex color string mapped to the given latency value."""
    if ms <= LATENCY_GOOD_MS:
        return LATENCY_GOOD_COLOR
    if ms <= LATENCY_WARN_MS:
        return LATENCY_WARN_COLOR
    return LATENCY_BAD_COLOR


def _latency_badge(ms: float) -> str:
    """Return a short quality badge label for the given latency."""
    if ms <= LATENCY_GOOD_MS:
        return "GREAT"
    if ms <= LATENCY_WARN_MS:
        return "OK"
    if ms <= LATENCY_BAD_MS:
        return "HIGH"
    return "POOR"


class VolumeMeter(tk.Canvas):
    """Horizontal LED-style volume meter."""

    SEGMENTS = 20
    COLORS = {
        "green": "#22c55e",
        "yellow": "#eab308",
        "red": "#ef4444",
    }

    def __init__(self, parent: tk.Widget, width: int = 120, height: int = 10, **kwargs) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=DARK_SURFACE,
            bd=0,
            highlightthickness=0,
            **kwargs,
        )
        self._level: float = 0.0
        self._width = width
        self._height = height
        self._draw(0.0)

    def update_level(self, level: float) -> None:
        """Update the meter with a new RMS level (0.0 – 1.0)."""
        self._level = max(0.0, min(1.0, level))
        self._draw(self._level)

    def _draw(self, level: float) -> None:
        self.delete("all")
        seg_w = (self._width - (self.SEGMENTS - 1) * 2) / self.SEGMENTS
        active_count = int(level * self.SEGMENTS)

        for i in range(self.SEGMENTS):
            x0 = i * (seg_w + 2)
            x1 = x0 + seg_w
            y0, y1 = 0, self._height

            if i < active_count:
                ratio = i / self.SEGMENTS
                if ratio < 0.65:
                    color = self.COLORS["green"]
                elif ratio < 0.85:
                    color = self.COLORS["yellow"]
                else:
                    color = self.COLORS["red"]
            else:
                color = "#2d2d44"  # inactive segment

            self.create_rectangle(x0, y0, x1, y1, fill=color, outline="")


class StatusBar(tk.Frame):
    """
    Footer status bar showing:
    - Active/Inactive indicator
    - Processing latency in ms
    - Input volume meter
    - Output volume meter
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, bg=DARK_SURFACE, pady=6, **kwargs)
        self._active = False
        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """Update the active indicator."""
        self._active = active
        if active:
            self._status_dot.config(bg=ACTIVE_COLOR)
            self._status_label.config(text="● ACTIVE", fg=ACTIVE_COLOR)
        else:
            self._status_dot.config(bg=INACTIVE_COLOR)
            self._status_label.config(text="● INACTIVE", fg=INACTIVE_COLOR)

    def update_latency(self, latency_ms: float) -> None:
        """Update the displayed latency value with color coding."""
        color = _latency_color(latency_ms)
        badge = _latency_badge(latency_ms)
        self._latency_value.config(text=f"{latency_ms:.1f}", fg=color)
        self._latency_badge.config(text=badge, fg=color)
        self._latency_unit.config(fg=color)

    def update_input_level(self, level: float) -> None:
        """Update the input volume meter (0.0 – 1.0)."""
        self._input_meter.update_level(level)

    def update_output_level(self, level: float) -> None:
        """Update the output volume meter (0.0 – 1.0)."""
        self._output_meter.update_level(level)

    def set_message(self, message: str) -> None:
        """Display an arbitrary status message."""
        self._message_label.config(text=message)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Status indicator ──────────────────────────────────────────────────
        self._status_label = tk.Label(
            self,
            text="● INACTIVE",
            fg=INACTIVE_COLOR,
            bg=DARK_SURFACE,
            font=("Segoe UI", 9, "bold"),
        )
        self._status_label.pack(side=tk.LEFT, padx=(10, 12))

        # Hidden dot (used internally)
        self._status_dot = tk.Label(self, bg=INACTIVE_COLOR, width=1, height=1)

        # ── Latency display (large, color-coded) ──────────────────────────────
        latency_frame = tk.Frame(self, bg=DARK_SURFACE)
        latency_frame.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(
            latency_frame, text="LATENCY",
            fg=SUBTEXT_COLOR, bg=DARK_SURFACE,
            font=("Segoe UI", 7, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 4), pady=0)

        self._latency_value = tk.Label(
            latency_frame, text="--",
            fg=SUBTEXT_COLOR, bg=DARK_SURFACE,
            font=("Segoe UI", 16, "bold"),
        )
        self._latency_value.pack(side=tk.LEFT)

        self._latency_unit = tk.Label(
            latency_frame, text=" ms",
            fg=SUBTEXT_COLOR, bg=DARK_SURFACE,
            font=("Segoe UI", 9),
        )
        self._latency_unit.pack(side=tk.LEFT, anchor="s", pady=(0, 2))

        self._latency_badge = tk.Label(
            latency_frame, text="",
            fg=SUBTEXT_COLOR, bg=DARK_SURFACE,
            font=("Segoe UI", 7, "bold"),
            padx=4, pady=1,
        )
        self._latency_badge.pack(side=tk.LEFT, padx=(6, 0), anchor="c")

        # ── Volume meters ─────────────────────────────────────────────────────
        tk.Label(
            self, text="IN:", fg=SUBTEXT_COLOR, bg=DARK_SURFACE, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._input_meter = VolumeMeter(self, width=100, height=10)
        self._input_meter.pack(side=tk.LEFT, padx=(0, 12), pady=2)

        tk.Label(
            self, text="OUT:", fg=SUBTEXT_COLOR, bg=DARK_SURFACE, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._output_meter = VolumeMeter(self, width=100, height=10)
        self._output_meter.pack(side=tk.LEFT, padx=(0, 12), pady=2)

        # ── Message (right-aligned) ───────────────────────────────────────────
        self._message_label = tk.Label(
            self,
            text="",
            fg=SUBTEXT_COLOR,
            bg=DARK_SURFACE,
            font=("Segoe UI", 8),
            anchor="e",
        )
        self._message_label.pack(side=tk.RIGHT, padx=10)