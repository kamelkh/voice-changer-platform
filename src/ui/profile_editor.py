"""
Profile editor dialog.

Opens as a modal Toplevel window and allows the user to create, edit,
or duplicate a :class:`~src.profiles.profile.VoiceProfile`.  Changes
are committed via the :class:`~src.profiles.profile_manager.ProfileManager`
so the profile grid refreshes automatically.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Callable, Optional

from src.profiles.profile import VoiceProfile
from src.utils.constants import (
    ACTIVE_COLOR,
    CARD_BG,
    DARK_BG,
    DARK_SURFACE,
    INACTIVE_COLOR,
    SUBTEXT_COLOR,
    TEXT_COLOR,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.profiles.profile_manager import ProfileManager

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_FIELD_BG   = "#1e1e2e"
_FIELD_FG   = TEXT_COLOR
_LABEL_FG   = SUBTEXT_COLOR
_BORDER_CLR = "#3b3b5a"

_EMOJI_PALETTE = [
    "🎙️", "🎤", "🤖", "👾", "🧛", "🧜", "👹", "🎭",
    "🦁", "👺", "🎵", "🎸", "🌌", "🔥", "💎", "⚡",
    "🐉", "🎲", "🌊", "🎼", "🤡", "👽", "🎷", "🎺",
]

_PITCH_MIN, _PITCH_MAX = -12.0, 12.0
_FORMANT_MIN, _FORMANT_MAX = -6.0, 6.0
_REVERB_MIN, _REVERB_MAX = 0.0, 1.0
_GATE_MIN, _GATE_MAX = -60.0, 0.0
_GAIN_MIN, _GAIN_MAX = 0.0, 4.0
_DISGUISE_MIN, _DISGUISE_MAX = 0.0, 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_label(parent: tk.Widget, text: str) -> tk.Label:
    return tk.Label(parent, text=text, fg=_LABEL_FG, bg=DARK_BG,
                    font=("Segoe UI", 9), anchor="w")


def _make_entry(parent: tk.Widget, textvariable: tk.StringVar,
                width: int = 28) -> tk.Entry:
    return tk.Entry(
        parent,
        textvariable=textvariable,
        bg=_FIELD_BG,
        fg=_FIELD_FG,
        insertbackground=_FIELD_FG,
        relief=tk.FLAT,
        bd=4,
        width=width,
        font=("Segoe UI", 10),
    )


def _make_slider(parent: tk.Widget, variable: tk.DoubleVar,
                 from_: float, to: float,
                 resolution: float = 0.1) -> tuple[tk.Scale, tk.Label]:
    """Return (Scale, value_label) pair."""
    val_label = tk.Label(parent, text=f"{variable.get():.2f}",
                         fg=_FIELD_FG, bg=DARK_BG,
                         font=("Segoe UI", 9), width=6, anchor="e")

    def _on_move(val: str) -> None:
        val_label.config(text=f"{float(val):.2f}")

    scale = tk.Scale(
        parent,
        variable=variable,
        from_=from_, to=to,
        resolution=resolution,
        orient=tk.HORIZONTAL,
        bg=DARK_BG,
        fg=_FIELD_FG,
        troughcolor=_FIELD_BG,
        activebackground=ACTIVE_COLOR,
        highlightthickness=0,
        bd=0,
        showvalue=False,
        command=_on_move,
    )
    return scale, val_label


# ── Profile Editor Dialog ─────────────────────────────────────────────────────


class ProfileEditorDialog:
    """
    Modal dialog for creating or editing a :class:`VoiceProfile`.

    Parameters
    ----------
    parent:
        Parent Tk widget (the main window).
    profile_manager:
        Live profile manager instance.
    key:
        If given, loads the existing profile for editing.
        If *None*, opens an empty form for a new profile.
    on_saved:
        Callable fired with the new/edited profile key after saving.
    """

    def __init__(
        self,
        parent: tk.Widget,
        profile_manager: "ProfileManager",
        key: Optional[str] = None,
        on_saved: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._pm = profile_manager
        self._edit_key = key
        self._on_saved = on_saved

        # Load existing profile or start fresh
        if key and (existing := self._pm.get_profile(key)):
            self._profile = existing
            title = f"Edit Profile — {existing.name}"
        else:
            self._profile = VoiceProfile()
            title = "New Profile"

        # ── Window ────────────────────────────────────────────────────────────
        self._win = tk.Toplevel(parent)
        self._win.title(title)
        self._win.geometry("540x640")
        self._win.resizable(False, False)
        self._win.config(bg=DARK_BG)
        self._win.grab_set()   # modal

        self._build_vars()
        self._build_ui()

    # ── Variable initialisation ───────────────────────────────────────────────

    def _build_vars(self) -> None:
        p = self._profile
        self._var_name        = tk.StringVar(value=p.name)
        self._var_desc        = tk.StringVar(value=p.description)
        self._var_icon        = tk.StringVar(value=p.icon or "🎙️")
        self._var_pitch       = tk.DoubleVar(value=p.pitch_shift)
        self._var_formant     = tk.DoubleVar(value=p.formant_shift)
        self._var_reverb      = tk.DoubleVar(value=p.reverb_level)
        self._var_gate        = tk.DoubleVar(value=p.noise_gate_threshold)
        self._var_gain        = tk.DoubleVar(value=p.gain)
        self._var_disguise        = tk.DoubleVar(value=getattr(p, 'voice_disguise', 0.0))
        self._var_accent_dialect  = tk.StringVar(value=getattr(p, 'accent_dialect', 'none'))
        self._var_accent_intensity = tk.DoubleVar(value=getattr(p, 'accent_intensity', 0.0))
        self._var_use_ai      = tk.BooleanVar(value=p.use_ai)
        self._var_ai_model    = tk.StringVar(value=p.ai_model_path)
        self._var_ai_f0       = tk.StringVar(value=p.ai_f0_method)
        self._var_ai_index    = tk.DoubleVar(value=p.ai_index_rate)
        self._var_ai_pitch    = tk.IntVar(value=p.ai_pitch_shift)
        self._var_ai_protect  = tk.DoubleVar(value=p.ai_protect)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Scrollable frame
        outer = tk.Frame(self._win, bg=DARK_BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        canvas = tk.Canvas(outer, bg=DARK_BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=DARK_BG)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self._build_identity_section(inner)
        self._sep(inner)
        self._build_effects_section(inner)
        self._sep(inner)
        self._build_ai_section(inner)

        # ── Buttons ────────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self._win, bg=DARK_SURFACE, pady=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Button(
            btn_frame, text="💾  Save Profile",
            bg=ACTIVE_COLOR, fg="white",
            activebackground="#16a34a", activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, bd=0, padx=20, pady=8, cursor="hand2",
            command=self._save,
        ).pack(side=tk.LEFT, padx=(16, 8))

        if self._edit_key:
            tk.Button(
                btn_frame, text="Duplicate",
                bg=DARK_BG, fg=TEXT_COLOR,
                activebackground=CARD_BG,
                font=("Segoe UI", 10),
                relief=tk.FLAT, bd=0, padx=14, pady=8, cursor="hand2",
                command=self._duplicate,
            ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_frame, text="Cancel",
            bg=DARK_BG, fg=SUBTEXT_COLOR,
            activebackground=CARD_BG,
            font=("Segoe UI", 10),
            relief=tk.FLAT, bd=0, padx=14, pady=8, cursor="hand2",
            command=self._win.destroy,
        ).pack(side=tk.RIGHT, padx=16)

    # ── Section builders ──────────────────────────────────────────────────────

    def _sep(self, parent: tk.Widget) -> None:
        tk.Frame(parent, bg=_BORDER_CLR, height=1).pack(fill=tk.X, pady=10)

    def _section_title(self, parent: tk.Widget, text: str) -> None:
        tk.Label(parent, text=text, fg=ACTIVE_COLOR, bg=DARK_BG,
                 font=("Segoe UI", 10, "bold"), anchor="w"
                 ).pack(fill=tk.X, pady=(0, 6))

    def _row(self, parent: tk.Widget, label_text: str,
             widget_fn) -> None:
        row = tk.Frame(parent, bg=DARK_BG)
        row.pack(fill=tk.X, pady=3)
        _make_label(row, label_text).pack(side=tk.LEFT, width=130, anchor="w")
        widget = widget_fn(row)
        if widget:
            widget.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _slider_row(self, parent: tk.Widget, label: str,
                    var: tk.DoubleVar, lo: float, hi: float,
                    res: float = 0.1) -> None:
        row = tk.Frame(parent, bg=DARK_BG)
        row.pack(fill=tk.X, pady=3)
        _make_label(row, label).pack(side=tk.LEFT, width=130)
        scale, val_lbl = _make_slider(row, var, lo, hi, res)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        val_lbl.pack(side=tk.LEFT, padx=(6, 0))

    def _build_identity_section(self, parent: tk.Widget) -> None:
        self._section_title(parent, "Identity")

        # Name
        row = tk.Frame(parent, bg=DARK_BG)
        row.pack(fill=tk.X, pady=3)
        _make_label(row, "Profile name").pack(side=tk.LEFT, width=130)
        _make_entry(row, self._var_name).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Description
        row2 = tk.Frame(parent, bg=DARK_BG)
        row2.pack(fill=tk.X, pady=3)
        _make_label(row2, "Description").pack(side=tk.LEFT, width=130)
        _make_entry(row2, self._var_desc).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Icon picker
        icon_row = tk.Frame(parent, bg=DARK_BG)
        icon_row.pack(fill=tk.X, pady=3)
        _make_label(icon_row, "Icon").pack(side=tk.LEFT, width=130)

        self._icon_preview = tk.Label(
            icon_row, textvariable=self._var_icon,
            bg=CARD_BG, fg=TEXT_COLOR,
            font=("Segoe UI Emoji", 22), width=3,
        )
        self._icon_preview.pack(side=tk.LEFT, padx=(0, 8))

        palette_frame = tk.Frame(icon_row, bg=DARK_BG)
        palette_frame.pack(side=tk.LEFT)
        for i, emoji in enumerate(_EMOJI_PALETTE):
            btn = tk.Button(
                palette_frame, text=emoji,
                bg=DARK_BG, fg=TEXT_COLOR,
                activebackground=CARD_BG,
                relief=tk.FLAT, bd=0,
                font=("Segoe UI Emoji", 14),
                cursor="hand2",
                command=lambda e=emoji: self._var_icon.set(e),
            )
            btn.grid(row=i // 8, column=i % 8, padx=1, pady=1)

    def _build_effects_section(self, parent: tk.Widget) -> None:
        self._section_title(parent, "Effects")
        self._slider_row(parent, "Pitch shift (st)", self._var_pitch,
                         _PITCH_MIN, _PITCH_MAX, 0.5)
        self._slider_row(parent, "Formant shift (st)", self._var_formant,
                         _FORMANT_MIN, _FORMANT_MAX, 0.25)
        self._slider_row(parent, "Reverb (wet)", self._var_reverb,
                         _REVERB_MIN, _REVERB_MAX, 0.05)
        self._slider_row(parent, "Noise gate (dB)", self._var_gate,
                         _GATE_MIN, _GATE_MAX, 1.0)
        self._slider_row(parent, "Gain (×)", self._var_gain,
                         _GAIN_MIN, _GAIN_MAX, 0.05)
        self._slider_row(parent, "🔒 Voice Disguise", self._var_disguise,
                         _DISGUISE_MIN, _DISGUISE_MAX, 0.05)

        # ── Accent dialect ────────────────────────────────────────────────
        dialect_row = tk.Frame(parent, bg=DARK_BG)
        dialect_row.pack(fill=tk.X, pady=3)
        _make_label(dialect_row, "🗣️ Dialect accent").pack(side=tk.LEFT, width=130)
        ttk.Combobox(
            dialect_row,
            textvariable=self._var_accent_dialect,
            values=["none", "palestinian", "syrian", "lebanese", "egyptian"],
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT)

        self._slider_row(parent, "Accent intensity", self._var_accent_intensity,
                         0.0, 1.0, 0.05)

    def _build_ai_section(self, parent: tk.Widget) -> None:
        self._section_title(parent, "AI Voice Conversion (RVC)")

        ai_chk = tk.Checkbutton(
            parent, text="Enable AI voice conversion",
            variable=self._var_use_ai,
            bg=DARK_BG, fg=TEXT_COLOR,
            activebackground=DARK_BG,
            selectcolor=_FIELD_BG,
            font=("Segoe UI", 9),
        )
        ai_chk.pack(anchor="w", pady=(0, 6))

        # Model path
        mp_row = tk.Frame(parent, bg=DARK_BG)
        mp_row.pack(fill=tk.X, pady=3)
        _make_label(mp_row, "Model (.pth path)").pack(side=tk.LEFT, width=130)
        _make_entry(mp_row, self._var_ai_model).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # F0 method
        f0_row = tk.Frame(parent, bg=DARK_BG)
        f0_row.pack(fill=tk.X, pady=3)
        _make_label(f0_row, "F0 method").pack(side=tk.LEFT, width=130)
        f0_cb = ttk.Combobox(f0_row, textvariable=self._var_ai_f0,
                             values=["rmvpe", "crepe", "harvest", "pm", "dio"],
                             state="readonly", width=14)
        f0_cb.pack(side=tk.LEFT)

        self._slider_row(parent, "Index rate", self._var_ai_index, 0.0, 1.0, 0.05)
        self._slider_row(parent, "Pitch (semitones)",
                         tk.DoubleVar(value=float(self._var_ai_pitch.get())),
                         -12.0, 12.0, 1.0)
        self._slider_row(parent, "Protect", self._var_ai_protect, 0.0, 0.5, 0.01)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _collect(self) -> Optional[VoiceProfile]:
        """Collect UI values into a VoiceProfile.  Returns None if invalid."""
        name = self._var_name.get().strip()
        if not name:
            messagebox.showerror("Validation Error", "Profile name cannot be empty.",
                                 parent=self._win)
            return None
        return VoiceProfile(
            name=name,
            description=self._var_desc.get().strip(),
            icon=self._var_icon.get(),
            pitch_shift=round(self._var_pitch.get(), 2),
            formant_shift=round(self._var_formant.get(), 2),
            reverb_level=round(self._var_reverb.get(), 3),
            noise_gate_threshold=round(self._var_gate.get(), 1),
            gain=round(self._var_gain.get(), 3),
            voice_disguise=round(self._var_disguise.get(), 3),
            accent_dialect=self._var_accent_dialect.get(),
            accent_intensity=round(self._var_accent_intensity.get(), 3),
            use_ai=self._var_use_ai.get(),
            ai_model_path=self._var_ai_model.get().strip(),
            ai_f0_method=self._var_ai_f0.get(),
            ai_index_rate=round(self._var_ai_index.get(), 3),
            ai_pitch_shift=self._var_ai_pitch.get(),
            ai_protect=round(self._var_ai_protect.get(), 3),
        )

    def _save(self) -> None:
        profile = self._collect()
        if profile is None:
            return

        # Derive a key from the name (slug)
        key = profile.name.lower().replace(" ", "_")
        if self._edit_key:
            # Rename if the name changed
            if self._edit_key != key:
                self._pm.delete_profile(self._edit_key)
            key = self._edit_key if self._edit_key == key else key

        self._pm.save_profile(key, profile)
        logger.info("Profile saved: %s", key)

        if self._on_saved:
            self._on_saved(key)

        self._win.destroy()

    def _duplicate(self) -> None:
        profile = self._collect()
        if profile is None:
            return
        profile.name += " (copy)"
        new_key = profile.name.lower().replace(" ", "_")
        self._pm.save_profile(new_key, profile)
        logger.info("Profile duplicated as: %s", new_key)
        if self._on_saved:
            self._on_saved(new_key)
        self._win.destroy()
