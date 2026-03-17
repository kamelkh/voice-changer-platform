"""
Device selector widget – dropdown pair for input and output device selection.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from src.audio.devices import AudioDevice, AudioDeviceManager
from src.utils.constants import (
    DARK_BG,
    DARK_SURFACE,
    SUBTEXT_COLOR,
    TEXT_COLOR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DeviceChangeCallback = Callable[[Optional[AudioDevice]], None]


class DeviceSelector(tk.Frame):
    """
    Two labelled dropdowns (Input / Output) for audio device selection.

    Example::

        selector = DeviceSelector(parent, device_manager)
        selector.pack(fill=tk.X, padx=12, pady=8)
        selector.on_input_change(lambda dev: stream.set_input_device(dev.index))
        selector.on_output_change(lambda dev: stream.set_output_device(dev.index))
    """

    def __init__(
        self,
        parent: tk.Widget,
        device_manager: AudioDeviceManager,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=DARK_BG, **kwargs)
        self._dm = device_manager
        self._input_callbacks: list[DeviceChangeCallback] = []
        self._output_callbacks: list[DeviceChangeCallback] = []

        self._input_var = tk.StringVar()
        self._output_var = tk.StringVar()

        self._build_ui()
        self.refresh()

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-query devices and repopulate dropdowns."""
        self._dm.refresh()
        self._populate_input_combo()
        self._populate_output_combo()

    def on_input_change(self, cb: DeviceChangeCallback) -> None:
        """Register a callback for input device changes."""
        self._input_callbacks.append(cb)

    def on_output_change(self, cb: DeviceChangeCallback) -> None:
        """Register a callback for output device changes."""
        self._output_callbacks.append(cb)

    def select_input_by_index(self, index: int) -> None:
        """Programmatically select input device by PortAudio index."""
        dev = self._dm.find_device_by_index(index)
        if dev:
            self._input_var.set(self._device_label(dev))

    def select_output_by_index(self, index: int) -> None:
        """Programmatically select output device by PortAudio index."""
        dev = self._dm.find_device_by_index(index)
        if dev:
            self._output_var.set(self._device_label(dev))

    def get_selected_input(self) -> Optional[AudioDevice]:
        """Return the currently selected input device, or *None*."""
        return self._find_by_label(self._input_var.get(), input_only=True)

    def get_selected_output(self) -> Optional[AudioDevice]:
        """Return the currently selected output device, or *None*."""
        return self._find_by_label(self._output_var.get(), input_only=False)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure(
            "Device.TCombobox",
            fieldbackground=DARK_SURFACE,
            background=DARK_SURFACE,
            foreground=TEXT_COLOR,
            selectbackground=DARK_SURFACE,
            selectforeground=TEXT_COLOR,
        )

        # Input row
        input_frame = tk.Frame(self, bg=DARK_BG)
        input_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            input_frame,
            text="🎤  Input Device",
            fg=SUBTEXT_COLOR,
            bg=DARK_BG,
            font=("Segoe UI", 9),
            width=16,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._input_combo = ttk.Combobox(
            input_frame,
            textvariable=self._input_var,
            state="readonly",
            style="Device.TCombobox",
            font=("Segoe UI", 9),
        )
        self._input_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._input_combo.bind("<<ComboboxSelected>>", self._on_input_selected)

        tk.Button(
            input_frame,
            text="↺",
            command=self.refresh,
            bg=DARK_SURFACE,
            fg=TEXT_COLOR,
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            cursor="hand2",
            padx=6,
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Output row
        output_frame = tk.Frame(self, bg=DARK_BG)
        output_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            output_frame,
            text="🔊  Output Device",
            fg=SUBTEXT_COLOR,
            bg=DARK_BG,
            font=("Segoe UI", 9),
            width=16,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._output_combo = ttk.Combobox(
            output_frame,
            textvariable=self._output_var,
            state="readonly",
            style="Device.TCombobox",
            font=("Segoe UI", 9),
        )
        self._output_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._output_combo.bind("<<ComboboxSelected>>", self._on_output_selected)

    def _populate_input_combo(self) -> None:
        devs = self._dm.get_input_devices()
        labels = [self._device_label(d) for d in devs]
        self._input_combo["values"] = labels
        if labels and not self._input_var.get():
            # Auto-select system default
            default = self._dm.get_default_input_device()
            if default:
                self._input_var.set(self._device_label(default))
            else:
                self._input_var.set(labels[0])

    def _populate_output_combo(self) -> None:
        devs = self._dm.get_output_devices()
        labels = [self._device_label(d) for d in devs]
        self._output_combo["values"] = labels
        if labels and not self._output_var.get():
            # Auto-select VB-Cable if available
            vb = self._dm.detect_vbcable_output()
            if vb:
                self._output_var.set(self._device_label(vb))
            else:
                default = self._dm.get_default_output_device()
                if default:
                    self._output_var.set(self._device_label(default))

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_input_selected(self, _event=None) -> None:
        dev = self.get_selected_input()
        logger.info("Input device selected: %s", dev)
        for cb in self._input_callbacks:
            try:
                cb(dev)
            except Exception as exc:
                logger.error("Input change callback error: %s", exc)

    def _on_output_selected(self, _event=None) -> None:
        dev = self.get_selected_output()
        logger.info("Output device selected: %s", dev)
        for cb in self._output_callbacks:
            try:
                cb(dev)
            except Exception as exc:
                logger.error("Output change callback error: %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _device_label(dev: AudioDevice) -> str:
        return f"[{dev.index}] {dev.name}"

    def _find_by_label(self, label: str, input_only: bool) -> Optional[AudioDevice]:
        devices = (
            self._dm.get_input_devices() if input_only else self._dm.get_output_devices()
        )
        for dev in devices:
            if self._device_label(dev) == label:
                return dev
        return None
