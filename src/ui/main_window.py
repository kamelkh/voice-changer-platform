"""
Main application window.

Assembles device selector, profile grid, effect controls,
toggle button and status bar into a single dark-themed Tkinter window.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Optional

from src.ui.device_selector import DeviceSelector
from src.ui.effect_controls import EffectControls
from src.ui.profile_editor import ProfileEditorDialog
from src.ui.profile_selector import ProfileSelector
from src.ui.status_bar import StatusBar
from src.utils.constants import (
    ACTIVE_COLOR,
    APP_TITLE,
    APP_VERSION,
    DARK_ACCENT,
    DARK_BG,
    DARK_SURFACE,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    INACTIVE_COLOR,
    LATENCY_PRESETS,
    SUBTEXT_COLOR,
    TEXT_COLOR,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.app import VoiceChangerApp

logger = get_logger(__name__)

# How often (ms) the UI polls the stream for level / latency updates
_UPDATE_INTERVAL_MS = 80


class MainWindow:
    """
    Tkinter main window for the Voice Changer Platform.

    The window owns the Tk root and drives the periodic UI refresh loop.
    All business logic is delegated to :class:`~src.app.VoiceChangerApp`.
    """

    def __init__(self, app: "VoiceChangerApp") -> None:
        self._app = app
        self._active = False

        # ── Root window setup ─────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title(f"{APP_TITLE}  v{APP_VERSION}")
        self.root.geometry(f"{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}")
        self.root.minsize(700, 520)
        self.root.config(bg=DARK_BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._set_window_icon()
        self._build_menu()
        self._build_body()
        self._schedule_update()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the Tkinter event loop (blocks until window is closed)."""
        self.root.mainloop()

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root, bg=DARK_SURFACE, fg=TEXT_COLOR, tearoff=0)

        # File menu
        file_menu = tk.Menu(menubar, bg=DARK_SURFACE, fg=TEXT_COLOR, tearoff=0)
        file_menu.add_command(label="Import Profile…", command=self._import_profile)
        file_menu.add_command(label="Export Profile…", command=self._export_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        # Settings menu
        settings_menu = tk.Menu(menubar, bg=DARK_SURFACE, fg=TEXT_COLOR, tearoff=0)
        settings_menu.add_command(label="Open Settings Folder", command=self._open_settings_folder)
        settings_menu.add_command(label="Open Models Folder", command=self._open_models_folder)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu
        help_menu = tk.Menu(menubar, bg=DARK_SURFACE, fg=TEXT_COLOR, tearoff=0)
        help_menu.add_command(label="How to Use with WhatsApp / Zoom / Discord", command=self._show_usage_help)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    # ── Body ──────────────────────────────────────────────────────────────────

    def _build_body(self) -> None:
        # ── Device selector (top) ─────────────────────────────────────────────
        device_frame = tk.Frame(self.root, bg=DARK_BG, pady=8)
        device_frame.pack(fill=tk.X, padx=16)

        self._device_selector = DeviceSelector(
            device_frame, device_manager=self._app.device_manager
        )
        self._device_selector.pack(fill=tk.X)
        self._device_selector.on_input_change(
            lambda dev: self._app.set_input_device(dev.index if dev else None)
        )
        self._device_selector.on_output_change(
            lambda dev: self._app.set_output_device(dev.index if dev else None)
        )

        tk.Frame(self.root, bg=DARK_SURFACE, height=1).pack(fill=tk.X, padx=0)

        # ── Profile selector (middle) ──────────────────────────────────────────
        self._profile_selector = ProfileSelector(
            self.root, profile_manager=self._app.profile_manager
        )
        self._profile_selector.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)
        self._profile_selector.on_profile_select(self._on_profile_selected)
        self._profile_selector.on_profile_new(self._on_new_profile)
        self._profile_selector.on_profile_edit(self._on_edit_profile)
        self._profile_selector.on_profile_delete(self._on_delete_profile)

        tk.Frame(self.root, bg=DARK_SURFACE, height=1).pack(fill=tk.X)

        # ── Bottom panel (effects + toggle + latency preset) ───────────────────
        bottom_frame = tk.Frame(self.root, bg=DARK_BG)
        bottom_frame.pack(fill=tk.X, padx=16, pady=6)

        # Effect sliders
        self._effect_controls = EffectControls(
            bottom_frame, on_change=self._on_effect_changed
        )
        self._effect_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel: toggle + latency preset selector
        right_panel = tk.Frame(bottom_frame, bg=DARK_BG)
        right_panel.pack(side=tk.RIGHT, padx=(12, 0))

        self._toggle_btn = tk.Button(
            right_panel,
            text="▶  START",
            font=("Segoe UI", 13, "bold"),
            bg=ACTIVE_COLOR,
            fg="white",
            activebackground="#16a34a",
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            padx=28,
            pady=18,
            cursor="hand2",
            command=self._toggle_active,
        )
        self._toggle_btn.pack(pady=(8, 4))

        # Monitor button (hear your own processed voice)
        self._monitor_btn = tk.Button(
            right_panel,
            text="🎧  Monitor OFF",
            font=("Segoe UI", 10),
            bg=DARK_SURFACE,
            fg=SUBTEXT_COLOR,
            activebackground="#1e3a5f",
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._toggle_monitor,
        )
        self._monitor_btn.pack(pady=(4, 4))

        # Latency preset selector
        preset_frame = tk.Frame(right_panel, bg=DARK_BG)
        preset_frame.pack(pady=(0, 4))

        tk.Label(
            preset_frame, text="Latency preset:",
            fg=SUBTEXT_COLOR, bg=DARK_BG,
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, padx=(0, 6))

        self._preset_var = tk.StringVar(value="balanced")
        preset_cb = ttk.Combobox(
            preset_frame,
            textvariable=self._preset_var,
            values=list(LATENCY_PRESETS.keys()),
            state="readonly",
            width=8,
        )
        preset_cb.pack(side=tk.LEFT)
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_changed)

        # AI model indicator
        self._ai_label = tk.Label(
            right_panel,
            text="",
            fg="#a78bfa",
            bg=DARK_BG,
            font=("Segoe UI", 8),
        )
        self._ai_label.pack()

        tk.Frame(self.root, bg=DARK_SURFACE, height=1).pack(fill=tk.X)

        # ── Status bar (bottom) ────────────────────────────────────────────────
        self._status_bar = StatusBar(self.root)
        self._status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Periodic update ───────────────────────────────────────────────────────

    def _schedule_update(self) -> None:
        """Schedule the next UI refresh tick."""
        self.root.after(_UPDATE_INTERVAL_MS, self._update_ui)

    def _update_ui(self) -> None:
        """Refresh meters and latency display from the audio stream."""
        try:
            stream = self._app.stream
            if stream and stream.is_running:
                self._status_bar.update_latency(stream.latency_ms)
                self._status_bar.update_input_level(stream.input_level * 8)   # scale for visibility
                self._status_bar.update_output_level(stream.output_level * 8)
            else:
                self._status_bar.update_input_level(0)
                self._status_bar.update_output_level(0)

            # Update monitor button to reflect current state
            if self._app.monitor_enabled:
                state = self._app.monitor_state
                if state == "recording":
                    self._monitor_btn.config(
                        text="🔴  Recording...",
                        bg="#dc2626", fg="white",
                    )
                elif state == "playing":
                    self._monitor_btn.config(
                        text="🔊  Playing...",
                        bg="#16a34a", fg="white",
                    )
                else:
                    self._monitor_btn.config(
                        text="🎧  Monitor ON — Speak!",
                        bg="#7c3aed", fg="white",
                    )
        except Exception as exc:
            logger.debug("UI update error: %s", exc)
        finally:
            self._schedule_update()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _toggle_active(self) -> None:
        """Start or stop the voice changing pipeline."""
        self._active = not self._active
        if self._active:
            ok = self._app.start()
            if not ok:
                self._active = False
                messagebox.showerror(
                    "Cannot Start",
                    "Failed to start audio stream.\n\n"
                    "Check that your input/output devices are available.",
                )
                return
            self._toggle_btn.config(
                text="■  STOP", bg=INACTIVE_COLOR, activebackground="#b91c1c"
            )
            self._status_bar.set_active(True)
            self._status_bar.set_message("Voice changer active")
        else:
            self._app.stop()
            self._toggle_btn.config(
                text="▶  START", bg=ACTIVE_COLOR, activebackground="#16a34a"
            )
            self._status_bar.set_active(False)
            self._status_bar.set_message("Voice changer stopped")

    def _toggle_monitor(self) -> None:
        """Toggle voice monitoring (record-then-playback through headphones)."""
        enabled = self._app.toggle_monitor()
        if enabled:
            self._monitor_btn.config(
                text="🎧  Monitor ON — Speak!",
                bg="#7c3aed",
                fg="white",
                activebackground="#6d28d9",
            )
            self._status_bar.set_message("🎧 Monitor ON — speak, then wait to hear playback")
        else:
            self._monitor_btn.config(
                text="🎧  Monitor OFF",
                bg=DARK_SURFACE,
                fg=SUBTEXT_COLOR,
                activebackground="#1e3a5f",
            )
            self._status_bar.set_message("Monitor OFF")

    def _on_profile_selected(self, key: str, profile) -> None:
        """Called when a profile card is clicked."""
        logger.info("Profile activated in UI: %s", key)
        self._effect_controls.load_from_profile(profile)
        self._app.apply_profile(profile)

        # Show AI indicator
        if profile.use_ai and profile.ai_model_path:
            self._ai_label.config(text=f"🤖 AI: {profile.ai_model_path}")
        else:
            self._ai_label.config(text="")

    def _on_effect_changed(self, param_name: str, value: float) -> None:
        """Called when an effect slider is moved."""
        self._app.update_effect_param(param_name, value)

    # ── Profile CRUD handlers ─────────────────────────────────────────────────

    def _on_new_profile(self) -> None:
        ProfileEditorDialog(
            parent=self.root,
            profile_manager=self._app.profile_manager,
            key=None,
            on_saved=self._after_profile_saved,
        )

    def _on_edit_profile(self, key: str) -> None:
        ProfileEditorDialog(
            parent=self.root,
            profile_manager=self._app.profile_manager,
            key=key,
            on_saved=self._after_profile_saved,
        )

    def _on_delete_profile(self, key: str) -> None:
        profile = self._app.profile_manager.get_profile(key)
        name = profile.name if profile else key
        if messagebox.askyesno(
            "Delete Profile",
            f"Delete profile '{name}'?\n\nThis cannot be undone.",
            parent=self.root,
        ):
            self._app.profile_manager.delete_profile(key)
            self._profile_selector.load_profiles()
            self._status_bar.set_message(f"Profile '{name}' deleted.")

    def _after_profile_saved(self, key: str) -> None:
        """Refresh the grid after a create/edit."""
        self._profile_selector.load_profiles()
        self._profile_selector.select_profile(key)
        self._status_bar.set_message(f"Profile '{key}' saved.")

    def _on_preset_changed(self, _event=None) -> None:
        """Apply a latency preset – restarts the stream if active."""
        preset = self._preset_var.get()
        self._app.apply_latency_preset(preset)
        self._status_bar.set_message(f"Latency preset: {preset}")



    def _import_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="Import Profile",
            filetypes=[("JSON Profile", "*.json"), ("All Files", "*.*")],
        )
        if path:
            key = self._app.profile_manager.import_profile(path)
            if key:
                self._profile_selector.load_profiles()
                messagebox.showinfo("Import Successful", f"Profile '{key}' imported.")
            else:
                messagebox.showerror("Import Failed", "Could not import the selected file.")

    def _export_profile(self) -> None:
        active_key = self._app.profile_manager.active_key
        if not active_key:
            messagebox.showwarning("No Profile Selected", "Select a profile first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Profile",
            defaultextension=".json",
            initialfile=f"{active_key}.json",
            filetypes=[("JSON Profile", "*.json")],
        )
        if path:
            ok = self._app.profile_manager.export_profile(active_key, path)
            if ok:
                messagebox.showinfo("Export Successful", f"Profile saved to:\n{path}")
            else:
                messagebox.showerror("Export Failed", "Could not export the profile.")

    def _open_settings_folder(self) -> None:
        import subprocess  # noqa: PLC0415
        import os  # noqa: PLC0415
        from src.utils.constants import CONFIG_DIR  # noqa: PLC0415
        CONFIG_DIR.mkdir(exist_ok=True)
        subprocess.Popen(f'explorer "{CONFIG_DIR}"')

    def _open_models_folder(self) -> None:
        import subprocess  # noqa: PLC0415
        from src.utils.constants import MODELS_DIR  # noqa: PLC0415
        MODELS_DIR.mkdir(exist_ok=True)
        subprocess.Popen(f'explorer "{MODELS_DIR}"')

    def _show_usage_help(self) -> None:
        messagebox.showinfo(
            "How to Use",
            "1. Install VB-Audio Virtual Cable.\n"
            "   https://vb-audio.com/Cable/\n\n"
            "2. Select your physical microphone as INPUT.\n\n"
            "3. Select 'CABLE Input (VB-Audio)' as OUTPUT.\n\n"
            "4. Click START.\n\n"
            "5. In WhatsApp Web / Zoom / Discord:\n"
            "   → Set microphone to: CABLE Output (VB-Audio)\n\n"
            "Your converted voice will be heard by others.",
        )

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            f"{APP_TITLE}\nVersion {APP_VERSION}\n\n"
            "Real-time AI voice conversion using RVC.\n"
            "GPU-accelerated with PyTorch + CUDA.\n\n"
            "MIT License",
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Gracefully shut down the app and close the window."""
        logger.info("Window close requested.")
        if self._active:
            self._app.stop()
        self._app.shutdown()
        self.root.destroy()

    def _set_window_icon(self) -> None:
        """Set a simple window icon if possible."""
        try:
            # Try to set a minimal icon bitmap
            icon_data = (
                "#define icon_width 16\n#define icon_height 16\n"
                "static unsigned char icon_bits[] = { 0x00,0x00,0x00,0x00 };"
            )
            # Silently ignore if icon setting fails
        except Exception:
            pass
