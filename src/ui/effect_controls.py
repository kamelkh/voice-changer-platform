"""
Effect controls widget – sliders for real-time audio effect parameters.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

from src.utils.constants import (
    DARK_BG,
    DARK_SURFACE,
    DARK_ACCENT,
    SUBTEXT_COLOR,
    TEXT_COLOR,
    MAX_FORMANT_SHIFT,
    MAX_GAIN,
    MAX_NOISE_GATE_DB,
    MAX_PITCH_SHIFT,
    MAX_REVERB,
    MIN_FORMANT_SHIFT,
    MIN_GAIN,
    MIN_NOISE_GATE_DB,
    MIN_PITCH_SHIFT,
    MIN_REVERB,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

SliderChangeCallback = Callable[[str, float], None]


class LabeledSlider(tk.Frame):
    """A horizontal slider with label and value display."""

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        from_: float,
        to: float,
        default: float,
        resolution: float,
        unit: str = "",
        on_change: Optional[Callable[[float], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=DARK_BG, **kwargs)
        self._unit = unit
        self._on_change = on_change
        self._var = tk.DoubleVar(value=default)

        # Label
        tk.Label(
            self,
            text=label,
            fg=SUBTEXT_COLOR,
            bg=DARK_BG,
            font=("Segoe UI", 9),
            width=16,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 6))

        # Slider
        self._slider = tk.Scale(
            self,
            variable=self._var,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg=DARK_BG,
            fg=TEXT_COLOR,
            troughcolor=DARK_SURFACE,
            activebackground=DARK_ACCENT,
            highlightthickness=0,
            bd=0,
            command=self._slider_moved,
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Value label
        self._val_label = tk.Label(
            self,
            text=self._format_value(default),
            fg=TEXT_COLOR,
            bg=DARK_BG,
            font=("Segoe UI", 9),
            width=7,
            anchor="e",
        )
        self._val_label.pack(side=tk.LEFT, padx=(6, 0))

    def _slider_moved(self, val_str: str) -> None:
        val = float(val_str)
        self._val_label.config(text=self._format_value(val))
        if self._on_change:
            try:
                self._on_change(val)
            except Exception as exc:
                logger.error("Slider callback error: %s", exc)

    def _format_value(self, val: float) -> str:
        if self._unit:
            return f"{val:+.1f} {self._unit}" if val != 0 else f"0 {self._unit}"
        return f"{val:.2f}"

    def get(self) -> float:
        """Return the current slider value."""
        return self._var.get()

    def set(self, value: float) -> None:
        """Programmatically set the slider value."""
        self._var.set(value)
        self._val_label.config(text=self._format_value(value))


class EffectControls(tk.Frame):
    """
    Panel of sliders controlling audio effect parameters.

    Each slider change fires the registered ``on_change`` callback with
    the parameter name and new value.
    """

    # Slider definitions: (attr_name, label, min, max, default, resolution, unit)
    _SLIDERS = [
        ("pitch_shift", "Pitch Shift", MIN_PITCH_SHIFT, MAX_PITCH_SHIFT, 0.0, 0.5, "st"),
        ("formant_shift", "Formant Shift", MIN_FORMANT_SHIFT, MAX_FORMANT_SHIFT, 0.0, 0.5, "st"),
        ("reverb_level", "Reverb", MIN_REVERB, MAX_REVERB, 0.0, 0.01, ""),
        ("noise_gate", "Noise Gate", MIN_NOISE_GATE_DB, MAX_NOISE_GATE_DB, -40.0, 1.0, "dB"),
        ("gain", "Volume / Gain", MIN_GAIN, MAX_GAIN, 1.0, 0.05, ""),
        ("voice_disguise", "🔒 Voice Disguise", 0.0, 1.0, 0.0, 0.05, ""),
    ]

    def __init__(
        self,
        parent: tk.Widget,
        on_change: Optional[SliderChangeCallback] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=DARK_BG, **kwargs)
        self._on_change = on_change
        self._sliders: dict[str, LabeledSlider] = {}

        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_param(self, name: str, value: float) -> None:
        """Set a specific slider by parameter name."""
        if name in self._sliders:
            self._sliders[name].set(value)

    def get_param(self, name: str) -> Optional[float]:
        """Return the current value of a named slider."""
        if name in self._sliders:
            return self._sliders[name].get()
        return None

    def load_from_profile(self, profile: "VoiceProfile") -> None:  # noqa: F821
        """Apply profile settings to all sliders."""
        self.set_param("pitch_shift", profile.pitch_shift)
        self.set_param("formant_shift", profile.formant_shift)
        self.set_param("reverb_level", profile.reverb_level)
        self.set_param("noise_gate", profile.noise_gate_threshold)
        self.set_param("gain", profile.gain)
        self.set_param("voice_disguise", getattr(profile, 'voice_disguise', 0.0))

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        header = tk.Label(
            self,
            text="Effect Controls",
            fg=TEXT_COLOR,
            bg=DARK_BG,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        header.pack(fill=tk.X, padx=4, pady=(6, 4))

        for attr, label, lo, hi, default, res, unit in self._SLIDERS:
            slider = LabeledSlider(
                self,
                label=label,
                from_=lo,
                to=hi,
                default=default,
                resolution=res,
                unit=unit,
                on_change=lambda v, a=attr: self._param_changed(a, v),
            )
            slider.pack(fill=tk.X, padx=8, pady=3)
            self._sliders[attr] = slider

    def _param_changed(self, name: str, value: float) -> None:
        if self._on_change:
            try:
                self._on_change(name, value)
            except Exception as exc:
                logger.error("EffectControls callback error: %s", exc)
