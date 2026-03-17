"""
Profile selector widget – a scrollable grid of clickable profile cards
with a CRUD action toolbar (New / Edit / Delete).
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

from src.profiles.profile import VoiceProfile
from src.profiles.profile_manager import ProfileManager
from src.utils.constants import (
    ACTIVE_COLOR,
    CARD_BG,
    CARD_SELECTED_BG,
    DARK_BG,
    DARK_SURFACE,
    INACTIVE_COLOR,
    SUBTEXT_COLOR,
    TEXT_COLOR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

ProfileSelectCallback = Callable[[str, VoiceProfile], None]
ProfileActionCallback = Callable[[str], None]


class ProfileCard(tk.Frame):
    """A single clickable profile card showing icon, name, and description."""

    def __init__(self, parent, key, profile, on_click, **kwargs):
        super().__init__(parent, bg=CARD_BG, relief=tk.FLAT, bd=0,
                         cursor="hand2", padx=12, pady=10, **kwargs)
        self.key = key
        self.profile = profile
        self._on_click = on_click
        self._selected = False
        self._build()
        self.bind("<Button-1>", self._click)
        for child in self.winfo_children():
            child.bind("<Button-1>", self._click)

    def _build(self):
        tk.Label(self, text=self.profile.icon or "🎙️", bg=CARD_BG, fg=TEXT_COLOR,
                 font=("Segoe UI Emoji", 22)).pack()
        tk.Label(self, text=self.profile.name, bg=CARD_BG, fg=TEXT_COLOR,
                 font=("Segoe UI", 10, "bold"), wraplength=110).pack(pady=(4,0))
        tk.Label(self, text=self.profile.description, bg=CARD_BG, fg=SUBTEXT_COLOR,
                 font=("Segoe UI", 8), wraplength=110).pack()
        if self.profile.use_ai:
            tk.Label(self, text="AI", bg="#7c3aed", fg="white",
                     font=("Segoe UI", 7, "bold"), padx=4, pady=1).pack(pady=(4,0))

    def set_selected(self, selected):
        self._selected = selected
        new_bg = CARD_SELECTED_BG if selected else CARD_BG
        self.config(bg=new_bg)
        for child in self.winfo_children():
            try:
                child.config(bg=new_bg)
            except Exception:
                pass

    def _click(self, _event=None):
        self._on_click(self.key)


class ProfileSelector(tk.Frame):
    """Scrollable grid of ProfileCard widgets with New/Edit/Delete toolbar."""

    CARDS_PER_ROW = 4

    def __init__(self, parent, profile_manager: ProfileManager, **kwargs):
        super().__init__(parent, bg=DARK_BG, **kwargs)
        self._pm = profile_manager
        self._select_callbacks: list[ProfileSelectCallback] = []
        self._new_callbacks: list[Callable[[], None]] = []
        self._edit_callbacks: list[ProfileActionCallback] = []
        self._delete_callbacks: list[ProfileActionCallback] = []
        self._cards: dict[str, ProfileCard] = {}
        self._selected_key: Optional[str] = None
        self._build_ui()
        self.load_profiles()

    # Public API
    def on_profile_select(self, cb: ProfileSelectCallback): self._select_callbacks.append(cb)
    def on_profile_new(self, cb: Callable[[], None]): self._new_callbacks.append(cb)
    def on_profile_edit(self, cb: ProfileActionCallback): self._edit_callbacks.append(cb)
    def on_profile_delete(self, cb: ProfileActionCallback): self._delete_callbacks.append(cb)

    @property
    def selected_key(self): return self._selected_key

    def load_profiles(self):
        for card in self._cards.values():
            card.destroy()
        self._cards.clear()
        for idx, (key, profile) in enumerate(self._pm.get_all_profiles()):
            row, col = divmod(idx, self.CARDS_PER_ROW)
            card = ProfileCard(self._grid_frame, key=key, profile=profile,
                               on_click=self._handle_card_click)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self._cards[key] = card
        for col in range(self.CARDS_PER_ROW):
            self._grid_frame.columnconfigure(col, weight=1)
        self._update_scroll_region()
        self._update_action_buttons()

    def select_profile(self, key):
        self._handle_card_click(key, fire_callbacks=False)

    def _build_ui(self):
        hdr = tk.Frame(self, bg=DARK_BG)
        hdr.pack(fill=tk.X, padx=4, pady=(4,6))
        tk.Label(hdr, text="Voice Profiles", fg=TEXT_COLOR, bg=DARK_BG,
                 font=("Segoe UI",11,"bold"), anchor="w").pack(side=tk.LEFT)

        self._btn_delete = tk.Button(hdr, text="✕  Delete",
            bg=DARK_SURFACE, fg=INACTIVE_COLOR,
            activebackground="#7f1d1d", activeforeground="white",
            relief=tk.FLAT, bd=0, padx=10, pady=4,
            font=("Segoe UI",9), cursor="hand2", command=self._on_delete_click)
        self._btn_delete.pack(side=tk.RIGHT, padx=(4,0))

        self._btn_edit = tk.Button(hdr, text="✎  Edit",
            bg=DARK_SURFACE, fg=SUBTEXT_COLOR,
            activebackground="#1e3a5f", activeforeground="white",
            relief=tk.FLAT, bd=0, padx=10, pady=4,
            font=("Segoe UI",9), cursor="hand2", command=self._on_edit_click)
        self._btn_edit.pack(side=tk.RIGHT, padx=(4,0))

        tk.Button(hdr, text="＋  New Profile",
            bg=ACTIVE_COLOR, fg="white",
            activebackground="#16a34a", activeforeground="white",
            relief=tk.FLAT, bd=0, padx=12, pady=4,
            font=("Segoe UI",9,"bold"), cursor="hand2",
            command=self._on_new_click).pack(side=tk.RIGHT, padx=(0,8))

        cf = tk.Frame(self, bg=DARK_BG)
        cf.pack(fill=tk.BOTH, expand=True)
        self._canvas = tk.Canvas(cf, bg=DARK_BG, bd=0, highlightthickness=0)
        sb = tk.Scrollbar(cf, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._grid_frame = tk.Frame(self._canvas, bg=DARK_BG)
        self._canvas_window = self._canvas.create_window((0,0), window=self._grid_frame, anchor="nw")
        self._grid_frame.bind("<Configure>", lambda _e: self._update_scroll_region())
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _update_action_buttons(self):
        state = tk.NORMAL if self._selected_key else tk.DISABLED
        self._btn_edit.config(state=state)
        self._btn_delete.config(state=state)

    def _update_scroll_region(self):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_new_click(self):
        for cb in self._new_callbacks:
            try: cb()
            except Exception as exc: logger.error("New callback error: %s", exc)

    def _on_edit_click(self):
        if not self._selected_key: return
        for cb in self._edit_callbacks:
            try: cb(self._selected_key)
            except Exception as exc: logger.error("Edit callback error: %s", exc)

    def _on_delete_click(self):
        if not self._selected_key: return
        for cb in self._delete_callbacks:
            try: cb(self._selected_key)
            except Exception as exc: logger.error("Delete callback error: %s", exc)

    def _handle_card_click(self, key, fire_callbacks=True):
        if self._selected_key and self._selected_key in self._cards:
            self._cards[self._selected_key].set_selected(False)
        if key in self._cards:
            self._cards[key].set_selected(True)
            self._selected_key = key
        self._update_action_buttons()
        profile = self._pm.get_profile(key)
        if profile is None:
            return
        logger.info("Profile selected: %s - %s", key, profile.name)
        self._pm.activate(key)
        if fire_callbacks:
            for cb in self._select_callbacks:
                try: cb(key, profile)
                except Exception as exc: logger.error("Select callback error: %s", exc)
