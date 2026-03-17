"""
Profile selector widget – a scrollable grid of clickable profile cards.
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
ProfileActionCallback = Callable[[str], None]   # key → None


class ProfileCard(tk.Frame):
    """A single clickable profile card showing icon, name, and description."""

    def __init__(
        self,
        parent: tk.Widget,
        key: str,
        profile: VoiceProfile,
        on_click: Callable[[str], None],
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            bg=CARD_BG,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            padx=12,
            pady=10,
            **kwargs,
        )
        self.key = key
        self.profile = profile
        self._on_click = on_click
        self._selected = False

        self._build()
        self.bind("<Button-1>", self._click)
        for child in self.winfo_children():
            child.bind("<Button-1>", self._click)

    def _build(self) -> None:
        icon_label = tk.Label(
            self,
            text=self.profile.icon or "🎙️",
            bg=CARD_BG,
            fg=TEXT_COLOR,
            font=("Segoe UI Emoji", 22),
        )
        icon_label.pack()

        name_label = tk.Label(
            self,
            text=self.profile.name,
            bg=CARD_BG,
            fg=TEXT_COLOR,
            font=("Segoe UI", 10, "bold"),
            wraplength=110,
        )
        name_label.pack(pady=(4, 0))

        desc_label = tk.Label(
            self,
            text=self.profile.description,
            bg=CARD_BG,
            fg=SUBTEXT_COLOR,
            font=("Segoe UI", 8),
            wraplength=110,
        )
        desc_label.pack()

        if self.profile.use_ai:
            ai_badge = tk.Label(
                self,
                text="AI",
                bg="#7c3aed",
                fg="white",
                font=("Segoe UI", 7, "bold"),
                padx=4,
                pady=1,
            )
            ai_badge.pack(pady=(4, 0))

    def set_selected(self, selected: bool) -> None:
        """Update visual state."""
        self._selected = selected
        new_bg = CARD_SELECTED_BG if selected else CARD_BG
        self.config(bg=new_bg)
        for child in self.winfo_children():
            try:
                child.config(bg=new_bg)
            except Exception:
                pass

    def _click(self, _event=None) -> None:
        self._on_click(self.key)


class ProfileSelector(tk.Frame):
    """
    Scrollable grid of :class:`ProfileCard` widgets with a CRUD toolbar.

    Extra callbacks:
    - ``on_profile_new``  – fired when the user clicks **＋ New**
    - ``on_profile_edit`` – fired with the selected key when **✎ Edit** is clicked
    - ``on_profile_delete`` – fired with the selected key when **✕ Delete** is clicked
    """

    CARDS_PER_ROW = 4

    def __init__(
        self,
        parent: tk.Widget,
        profile_manager: ProfileManager,
        **kwargs,
    ) -> None:
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

    # ── Public API ────────────────────────────────────────────────────────────

    def on_profile_select(self, cb: ProfileSelectCallback) -> None:
        """Register a callback fired when a card is clicked."""
        self._select_callbacks.append(cb)

    def on_profile_new(self, cb: Callable[[], None]) -> None:
        """Register a callback for the 'New Profile' button."""
        self._new_callbacks.append(cb)

    def on_profile_edit(self, cb: ProfileActionCallback) -> None:
        """Register a callback for the 'Edit' button (receives selected key)."""
        self._edit_callbacks.append(cb)

    def on_profile_delete(self, cb: ProfileActionCallback) -> None:
        """Register a callback for the 'Delete' button (receives selected key)."""
        self._delete_callbacks.append(cb)

    def load_profiles(self) -> None:
        """Rebuild the card grid from the profile manager."""
        # Clear existing cards
        for card in self._cards.values():
            card.destroy()
        self._cards.clear()

        profiles = self._pm.get_all_profiles()
        for idx, (key, profile) in enumerate(profiles):
            row, col = divmod(idx, self.CARDS_PER_ROW)
            card = ProfileCard(
                self._grid_frame,
                key=key,
                profile=profile,
                on_click=self._handle_card_click,
            )
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self._cards[key] = card

        # Configure column weights for uniform sizing
        for col in range(self.CARDS_PER_ROW):
            self._grid_frame.columnconfigure(col, weight=1)

        self._update_scroll_region()
        self._update_action_buttons()

    def select_profile(self, key: str) -> None:
        """Programmatically select a profile card."""
        self._handle_card_click(key, fire_callbacks=False)

    @property
    def selected_key(self) -> Optional[str]:
        return self._selected_key

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Header row: title + action buttons ───────────────────────────────
        header_frame = tk.Frame(self, bg=DARK_BG)
        header_frame.pack(fill=tk.X, padx=4, pady=(4, 6))

        tk.Label(
            header_frame,
            text="Voice Profiles",
            fg=TEXT_COLOR,
            bg=DARK_BG,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).pack(side=tk.LEFT)

        # Delete button (right side)
        self._btn_delete = tk.Button(
            header_frame,
            text="✕  Delete",
            bg=DARK_SURFACE,
            fg=INACTIVE_COLOR,
            activebackground="#7f1d1d",
            activeforeground="white",
            relief=tk.FLAT, bd=0,
            padx=10, pady=4,
            font=("Segoe UI", 9),
            cursor="hand2",
            command=self._on_delete_click,
        )
        self._btn_delete.pack(side=tk.RIGHT, padx=(4, 0))

        # Edit button
        self._btn_edit = tk.Button(
            header_frame,
            text="✎  Edit",
            bg=DARK_SURFACE,
            fg=SUBTEXT_COLOR,
            activebackground="#1e3a5f",
            activeforeground="white",
            relief=tk.FLAT, bd=0,
            padx=10, pady=4,
            font=("Segoe UI", 9),
            cursor="hand2",
            command=self._on_edit_click,
        )
        self._btn_edit.pack(side=tk.RIGHT, padx=(4, 0))

        # New button
        tk.Button(
            header_frame,
            text="＋  New Profile",
            bg=ACTIVE_COLOR,
            fg="white",
            activebackground="#16a34a",
            activeforeground="white",
            relief=tk.FLAT, bd=0,
            padx=12, pady=4,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._on_new_click,
        ).pack(side=tk.RIGHT, padx=(0, 8))

        # ── Canvas + scrollbar ────────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=DARK_BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(canvas_frame, bg=DARK_BG, bd=0, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._grid_frame = tk.Frame(self._canvas, bg=DARK_BG)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._grid_frame, anchor="nw"
        )

        self._grid_frame.bind("<Configure>", lambda _e: self._update_scroll_region())
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _update_action_buttons(self) -> None:
        """Enable/disable Edit and Delete based on whether a card is selected."""
        has_sel = self._selected_key is not None
        state = tk.NORMAL if has_sel else tk.DISABLED
        self._btn_edit.config(state=state)
        self._btn_delete.config(state=state)

    def _update_scroll_region(self) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Toolbar click handlers ────────────────────────────────────────────────

    def _on_new_click(self) -> None:
        for cb in self._new_callbacks:
            try:
                cb()
            except Exception as exc:
                logger.error("New-profile callback error: %s", exc)

    def _on_edit_click(self) -> None:
        if not self._selected_key:
            return
        for cb in self._edit_callbacks:
            try:
                cb(self._selected_key)
            except Exception as exc:
                logger.error("Edit-profile callback error: %s", exc)

    def _on_delete_click(self) -> None:
        if not self._selected_key:
            return
        for cb in self._delete_callbacks:
            try:
                cb(self._selected_key)
            except Exception as exc:
                logger.error("Delete-profile callback error: %s", exc)

    # ── Event handling ────────────────────────────────────────────────────────

    def _handle_card_click(self, key: str, fire_callbacks: bool = True) -> None:
        # Deselect previous
        if self._selected_key and self._selected_key in self._cards:
            self._cards[self._selected_key].set_selected(False)

        # Select new
        if key in self._cards:
            self._cards[key].set_selected(True)
            self._selected_key = key

        self._update_action_buttons()

        profile = self._pm.get_profile(key)
        if profile is None:
            return

        logger.info("Profile selected: %s – %s", key, profile.name)
        self._pm.activate(key)

        if fire_callbacks:
            for cb in self._select_callbacks:
                try:
                    cb(key, profile)
                except Exception as exc:
                    logger.error("Profile select callback error: %s", exc)


    def __init__(
        self,
        parent: tk.Widget,
        profile_manager: ProfileManager,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=DARK_BG, **kwargs)
        self._pm = profile_manager
        self._select_callbacks: list[ProfileSelectCallback] = []
        self._cards: dict[str, ProfileCard] = {}
        self._selected_key: Optional[str] = None

        self._build_ui()
        self.load_profiles()

    # ── Public API ────────────────────────────────────────────────────────────

    def on_profile_select(self, cb: ProfileSelectCallback) -> None:
        """Register a callback fired when a card is clicked."""
        self._select_callbacks.append(cb)

    def load_profiles(self) -> None:
        """Rebuild the card grid from the profile manager."""
        # Clear existing cards
        for card in self._cards.values():
            card.destroy()
        self._cards.clear()

        profiles = self._pm.get_all_profiles()
        for idx, (key, profile) in enumerate(profiles):
            row, col = divmod(idx, self.CARDS_PER_ROW)
            card = ProfileCard(
                self._grid_frame,
                key=key,
                profile=profile,
                on_click=self._handle_card_click,
            )
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self._cards[key] = card

        # Configure column weights for uniform sizing
        for col in range(self.CARDS_PER_ROW):
            self._grid_frame.columnconfigure(col, weight=1)

        self._update_scroll_region()

    def select_profile(self, key: str) -> None:
        """Programmatically select a profile card."""
        self._handle_card_click(key, fire_callbacks=False)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Section header
        header = tk.Label(
            self,
            text="Voice Profiles",
            fg=TEXT_COLOR,
            bg=DARK_BG,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        header.pack(fill=tk.X, padx=4, pady=(4, 6))

        # Canvas + scrollbar container
        canvas_frame = tk.Frame(self, bg=DARK_BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(canvas_frame, bg=DARK_BG, bd=0, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._grid_frame = tk.Frame(self._canvas, bg=DARK_BG)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._grid_frame, anchor="nw"
        )

        self._grid_frame.bind("<Configure>", lambda _e: self._update_scroll_region())
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _update_scroll_region(self) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Event handling ────────────────────────────────────────────────────────

    def _handle_card_click(self, key: str, fire_callbacks: bool = True) -> None:
        # Deselect previous
        if self._selected_key and self._selected_key in self._cards:
            self._cards[self._selected_key].set_selected(False)

        # Select new
        if key in self._cards:
            self._cards[key].set_selected(True)
            self._selected_key = key

        profile = self._pm.get_profile(key)
        if profile is None:
            return

        logger.info("Profile selected: %s – %s", key, profile.name)
        self._pm.activate(key)

        if fire_callbacks:
            for cb in self._select_callbacks:
                try:
                    cb(key, profile)
                except Exception as exc:
                    logger.error("Profile select callback error: %s", exc)
