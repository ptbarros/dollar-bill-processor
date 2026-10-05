"""
Crop Setup Wizard — guided per-denomination crop-profile builder.

For users who find the Crop Manager daunting, this walks them through building a
crop profile one crop at a time, on a built-in sample bill of the denomination
they pick (so they can set everything up before they've scanned anything):

    Intro
      -> Pick a denomination + name the profile   (uses that sample bill)
      -> For each crop in turn: show it on the bill, let them place its box
         (seals/serials drag; thirds/full are automatic + an on/off toggle)
      -> "You can reorder / enable / disable these later" explainer
      -> Make another denomination profile, or done?
      -> Explain input/output folders, then ready to crop.

Seals and serials normally anchor to features the ($1-trained) detection model
finds. When a denomination has no anchor for a crop, the step falls back to a
plain hand-placed box saved as a fraction of the bill (see crop_geometry.set_fixed
and process_production._fixed_box_rect).

Built on the same pieces as the interactive Crop Manager: CropCanvas (drag),
crop_geometry (drag<->config), crop_samples (the sample bills), and the app's
crop-profile config. The wizard writes finished profiles into config.yaml and
hands the crop-tool window the input/output folders + active profile.
"""

from __future__ import annotations
import copy
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QFileDialog, QStackedWidget, QWidget, QApplication,
)
from PySide6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).parent.parent))

import crop_samples
import crop_geometry as cg
from gui.crop_canvas import CropCanvas

# The canonical crop list + friendly names (kept in sync with the Crop Manager).
from gui.crop_dialog import EbayCropDialog
_CROPS = EbayCropDialog.DEFAULT_CROPS

_DENOM_NUMBER = {
    'one_dollar': 1, 'two_dollar': 2, 'five_dollar': 5, 'ten_dollar': 10,
    'twenty_dollar': 20, 'fifty_dollar': 50, 'hundred_dollar': 100,
}


def _hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: gray; font-size: 11px;")
    return lbl


def _title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
    lbl.setWordWrap(True)
    return lbl


class CropProfileWizard(QDialog):
    """exec() it. On Accepted, read ``input_dir`` / ``output_dir`` and the profiles
    it saved (``saved_profiles`` names; it also writes them to config and sets the
    active one). ``ctx_factory(sample_dir)`` must return a CropPreviewContext (or
    None) for a sample folder; it's called when a denomination is chosen."""

    # Pages in the stack
    _INTRO, _DENOM, _CROP, _DONE, _FOLDERS, _READY = range(6)

    def __init__(self, ctx_factory, save_config, base_yolo_crops=None,
                 parent=None, initial_input="", initial_output="",
                 existing_profiles=None, folder_setup=True):
        super().__init__(parent)
        # folder_setup=False (main app): the wizard only BUILDS profiles -- skip the
        # input/output folder + "ready to crop a folder" pages, which are specific
        # to the standalone Crop Detective tool.
        self._folder_setup = folder_setup
        self.setWindowTitle("Crop Setup Wizard")
        self.setMinimumSize(900, 620)
        self._ctx_factory = ctx_factory
        self._save_config = save_config       # callable(profiles: dict, active: str)
        self._base_yolo = copy.deepcopy(base_yolo_crops or {})
        # Profiles already in config, so a denomination the user sets up again is
        # LOADED and OVERWRITTEN (edited) instead of piling up as "$20 (2)".
        self._existing = copy.deepcopy(existing_profiles or {})

        self.samples = crop_samples.discover()
        self.saved_profiles = {}              # name -> profile dict (also persisted)
        self.input_dir = initial_input
        self.output_dir = initial_output or str(Path.home() / "Crop Detective Output")

        # Per-profile working state (reset each denomination)
        self._sample = None
        self._ctx = None
        self._wcfg = {}
        self._enabled = {}
        self._crop_idx = 0
        self._canvas_side = None

        self._build()
        self._goto(self._INTRO)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self.stack.addWidget(self._page_intro())      # 0
        self.stack.addWidget(self._page_denom())       # 1
        self.stack.addWidget(self._page_crop())        # 2
        self.stack.addWidget(self._page_done())        # 3
        self.stack.addWidget(self._page_folders())     # 4
        self.stack.addWidget(self._page_ready())       # 5

        # Shared nav bar
        nav = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._on_back)
        nav.addWidget(self.back_btn)
        nav.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        nav.addWidget(self.cancel_btn)
        self.next_btn = QPushButton("Next")
        self.next_btn.setDefault(True)
        self.next_btn.clicked.connect(self._on_next)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)

    def _page_intro(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(_title("Welcome — let's set up your crops"))
        body = QLabel(
            "We'll build a <b>crop profile</b> for a denomination — the set of "
            "crops (seals, serials, and each third of the note) and where each one "
            "sits.<br><br>"
            "You don't need any scanned bills yet: we'll use a <b>sample bill</b> so "
            "you can place each crop and see it right away. You can make a profile "
            "for each denomination you handle."
        )
        body.setWordWrap(True); body.setTextFormat(Qt.RichText)
        v.addWidget(body); v.addStretch()
        return w

    def _page_denom(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(_title("Which denomination?"))
        v.addWidget(QLabel("Pick the bill you want to set up crops for. We'll use "
                           "our sample of it."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Denomination:"))
        self.denom_combo = QComboBox()
        for s in self.samples:
            self.denom_combo.addItem(s['label'], s['key'])
        self.denom_combo.currentIndexChanged.connect(self._sync_default_name)
        row.addWidget(self.denom_combo)
        row.addSpacing(20)
        row.addWidget(QLabel("Profile name:"))
        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_name_edited)
        self.name_edit.textChanged.connect(self._update_name_status)
        row.addWidget(self.name_edit, 1)
        v.addLayout(row)
        v.addWidget(_hint("The profile name is how you'll pick this setup later "
                          "(e.g. '$5', or '$5 tight'). You can rename it any time."))
        self.name_status = _hint("")
        v.addWidget(self.name_status)
        v.addStretch()
        self._sync_default_name()
        return w

    def _on_name_edited(self, _text):
        self._name_touched = True

    def _sync_default_name(self):
        if self.samples and not getattr(self, '_name_touched', False):
            self.name_edit.setText(self.denom_combo.currentData() and
                                   crop_samples.pretty_label(self.denom_combo.currentData()))
        self._update_name_status()

    def _update_name_status(self, *_):
        name = self.name_edit.text().strip()
        if name and (name in self._existing or name in self.saved_profiles):
            self.name_status.setText(
                f"↻ A profile named <b>{name}</b> already exists — its current crops "
                "will be loaded so you can edit them, and saving will update it.")
        elif name:
            self.name_status.setText(f"✚ New profile <b>{name}</b> will be created.")
        else:
            self.name_status.setText("")

    def _page_crop(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.crop_title = _title("")
        v.addWidget(self.crop_title)
        self.crop_hint = QLabel(""); self.crop_hint.setWordWrap(True)
        v.addWidget(self.crop_hint)
        self.canvas = CropCanvas()
        self.canvas.geometryChanged.connect(self._on_crop_drag)
        self.canvas.geometryLive.connect(self._update_crop_info)  # live size/pos readout
        v.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.enable_cb = QCheckBox("Include this crop")
        self.enable_cb.toggled.connect(self._on_enable_toggled)
        row.addWidget(self.enable_cb)
        row.addSpacing(16)
        self.fixed_cb = QCheckBox("Place at a fixed spot (don't auto-detect)")
        self.fixed_cb.setToolTip(
            "On: the crop stays exactly where you put it, ignoring the detector.\n"
            "Use this when the seal/serial isn't found reliably — e.g. non-$1 notes,\n"
            "where the ($1-trained) model may anchor in the wrong place.")
        self.fixed_cb.toggled.connect(self._on_fixed_toggled)
        row.addWidget(self.fixed_cb)
        row.addStretch()
        self.reset_btn = QPushButton("Reset this box")
        self.reset_btn.setToolTip("Put this crop's box back to the automatic position.")
        self.reset_btn.clicked.connect(self._on_reset_box)
        row.addWidget(self.reset_btn)
        v.addLayout(row)
        self.crop_info = _hint("")
        v.addWidget(self.crop_info)
        return w

    def _page_done(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(_title("Profile ready"))
        self.done_body = QLabel(); self.done_body.setWordWrap(True)
        self.done_body.setTextFormat(Qt.RichText)
        v.addWidget(self.done_body)
        row = QHBoxLayout()
        self.another_btn = QPushButton("Make another denomination profile")
        self.another_btn.clicked.connect(self._on_make_another)
        row.addWidget(self.another_btn)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()
        return w

    def _page_folders(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(_title("Where are your bills, and where should crops go?"))
        v.addWidget(QLabel("When you're ready to crop real bills, the tool needs two "
                           "folders. You can set these now or later in the main window."))
        # input
        v.addWidget(QLabel("<b>Folder with your scanned bills</b> (front and back):"))
        r1 = QHBoxLayout()
        self.in_edit = QLineEdit(self.input_dir)
        r1.addWidget(self.in_edit)
        b1 = QPushButton("Browse…"); b1.clicked.connect(lambda: self._browse(self.in_edit))
        r1.addWidget(b1); v.addLayout(r1)
        v.addWidget(_hint("Leave blank for now if you haven't scanned any yet."))
        # output
        v.addWidget(QLabel("<b>Folder to save the crops in:</b>"))
        r2 = QHBoxLayout()
        self.out_edit = QLineEdit(self.output_dir)
        r2.addWidget(self.out_edit)
        b2 = QPushButton("Browse…"); b2.clicked.connect(lambda: self._browse(self.out_edit))
        r2.addWidget(b2); v.addLayout(r2)
        v.addStretch()
        return w

    def _page_ready(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(_title("All set"))
        self.ready_body = QLabel(); self.ready_body.setWordWrap(True)
        self.ready_body.setTextFormat(Qt.RichText)
        v.addWidget(self.ready_body); v.addStretch()
        return w

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _goto(self, page):
        self._page = page
        self.stack.setCurrentIndex(page)
        self.back_btn.setEnabled(page not in (self._INTRO,))
        # Per-page button text + entry setup
        if page == self._CROP:
            self._enter_crop()
            last = self._crop_idx >= len(_CROPS) - 1
            self.next_btn.setText("Finish crops" if last else "Next crop ▸")
        elif page == self._DONE:
            self._enter_done()
            self.next_btn.setText("I'm done ▸" if self._folder_setup else "Finish")
        elif page == self._READY:
            self._enter_ready()
            self.next_btn.setText("Finish")
        elif page == self._FOLDERS:
            self.next_btn.setText("Next")
        else:
            self.next_btn.setText("Next")

    def _on_next(self):
        p = self._page
        if p == self._INTRO:
            self._goto(self._DENOM)
        elif p == self._DENOM:
            if not self._start_profile():
                return
            self._crop_idx = 0
            self._goto(self._CROP)
        elif p == self._CROP:
            if self._crop_idx >= len(_CROPS) - 1:
                self._finalize_profile()
                self._goto(self._DONE)
            else:
                self._crop_idx += 1
                self._goto(self._CROP)
        elif p == self._DONE:
            if self._folder_setup:
                self._goto(self._FOLDERS)
            else:
                self.accept()   # main app: profiles only, no folder step
        elif p == self._FOLDERS:
            self.input_dir = self.in_edit.text().strip()
            self.output_dir = self.out_edit.text().strip()
            self._goto(self._READY)
        elif p == self._READY:
            self.accept()

    def _on_back(self):
        p = self._page
        if p == self._DENOM:
            self._goto(self._INTRO)
        elif p == self._CROP:
            if self._crop_idx == 0:
                self._goto(self._DENOM)
            else:
                self._crop_idx -= 1
                self._goto(self._CROP)
        elif p == self._DONE:
            self._crop_idx = len(_CROPS) - 1
            self._goto(self._CROP)
        elif p == self._FOLDERS:
            self._goto(self._DONE)
        elif p == self._READY:
            self._goto(self._FOLDERS)

    def _on_make_another(self):
        self._goto(self._DENOM)

    # ------------------------------------------------------------------
    # Profile lifecycle
    # ------------------------------------------------------------------
    def _start_profile(self) -> bool:
        """Load the chosen denomination's sample + reset working state."""
        key = self.denom_combo.currentData()
        self._sample = crop_samples.get(key)
        if not self._sample:
            # No sample for the chosen denomination (or none bundled at all) ->
            # say so instead of silently doing nothing on Next.
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Sample bill not found",
                "The built-in sample bills weren't found, so the wizard can't set up "
                "crops.\n\nTry reinstalling the latest version; if it keeps happening, "
                "let us know.")
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._ctx = self._ctx_factory(self._sample['dir'])
        finally:
            QApplication.restoreOverrideCursor()
        if self._ctx is None:
            self.crop_hint.setText("Couldn't load the sample bill for preview.")
            return False
        self._profile_name = (self.name_edit.text().strip()
                              or self._sample['label'])
        # If a profile with THIS name already exists, load it so the user edits its
        # crops; otherwise start FRESH (empty yolo_crops) so every crop uses the
        # model's own detection for THIS bill -- never some other profile's boxes.
        # (Pipeline defaults, e.g. serial min 500px, fill in at crop time.)
        prior = self.saved_profiles.get(self._profile_name) \
            or self._existing.get(self._profile_name)
        if isinstance(prior, dict):
            self._wcfg = {'yolo_crops': copy.deepcopy(prior.get('yolo_crops', {}) or {})}
            order = {tuple(c) for c in (prior.get('crop_order') or [])}
            # Legacy profile with no crop_order -> treat all as enabled.
            self._enabled = {(c['side'], c['region']):
                             (not order or (c['side'], c['region']) in order)
                             for c in _CROPS}
        else:
            self._wcfg = {'yolo_crops': {}}
            self._enabled = {(c['side'], c['region']): True for c in _CROPS}
        self._canvas_side = None
        return True

    def _cur_crop(self):
        return _CROPS[self._crop_idx]

    def _region_key(self, side, region):
        return cg.region_config_key(side, region)

    def _crop_kind(self, region):
        if region in cg.THIRDS_REGIONS:
            return 'thirds'
        if region == 'full':
            return 'auto'
        return 'box'      # seal / serial_left / serial_right

    def _enter_crop(self):
        c = self._cur_crop()
        side, region, name = c['side'], c['region'], c['name']
        self.crop_title.setText(
            f"{self._profile_name}: {name}   (crop {self._crop_idx + 1} of {len(_CROPS)})")
        self.enable_cb.blockSignals(True)
        self.enable_cb.setChecked(self._enabled[(side, region)])
        self.enable_cb.blockSignals(False)

        if not self._ctx.has_side(side):
            self.canvas.set_bill(None)
            self.crop_hint.setText(f"(No {side} sample available.)")
            self.crop_info.setText("")
            self.reset_btn.setVisible(False)
            return

        kind = self._crop_kind(region)
        rect = self._ctx.render(side, region, self._wcfg)[1]

        # The fixed/anchor toggle only applies to box-mode (seal/serial) crops.
        self.fixed_cb.setVisible(kind == 'box')

        if kind == 'thirds':
            edges = cg.thirds_edges(region)
            if self._canvas_side != side:
                self.canvas.set_bill(self._ctx.imgs[side])
                self._canvas_side = side
            self.canvas.show_region(rect, others=[], editable=True,
                                    mode="edges", edges=edges)
            which = ("its right edge" if region == 'left'
                     else "its left edge" if region == 'right'
                     else "either side")
            self.crop_hint.setText(
                f"<b>{name}</b> splits the bill automatically. Drag {which} to overlap "
                "into the neighbouring crop (handy for a defect near the boundary).")
            self.reset_btn.setVisible(True)
        elif kind == 'auto':
            if self._canvas_side != side:
                self.canvas.set_bill(self._ctx.imgs[side])
                self._canvas_side = side
            self.canvas.show_region(rect, others=[], editable=False)
            self.crop_hint.setText(
                f"<b>{name}</b> is placed automatically — it follows the bill, so "
                "there's nothing to position. Use the checkbox to include it or not.")
            self.reset_btn.setVisible(False)
        else:  # 'box' (seal / serial)
            key = self._region_key(side, region)
            has_anchor = self._has_anchor(side, region, key)
            # No anchor at all -> must be a hand-placed fixed box.
            if not has_anchor and not cg.is_fixed(self._wcfg, key):
                self._set_default_fixed(side, key)
            is_fixed = cg.is_fixed(self._wcfg, key)
            rect = self._ctx.render(side, region, self._wcfg)[1]

            self.fixed_cb.blockSignals(True)
            self.fixed_cb.setChecked(is_fixed)
            # If there's nothing to anchor to, fixed is forced (can't uncheck).
            self.fixed_cb.setEnabled(has_anchor)
            self.fixed_cb.blockSignals(False)

            if self._canvas_side != side:
                self.canvas.set_bill(self._ctx.imgs[side])
                self._canvas_side = side
            self.canvas.show_region(rect, others=[], editable=True, mode="box")

            if is_fixed and not has_anchor:
                self.crop_hint.setText(
                    f"We couldn't find <b>{name}</b> automatically on the "
                    f"{self._sample['label']} note, so drag this box to where you "
                    "want the crop — it stays exactly there.")
            elif is_fixed:
                self.crop_hint.setText(
                    f"<b>{name}</b> is placed at a <b>fixed spot</b> — it stays exactly "
                    "where you put it. Drag/resize the box. (Untick the box to anchor "
                    "it to the detected feature instead.)")
            else:
                self.crop_hint.setText(
                    f"Drag the blue box to fine-tune <b>{name}</b>, or resize it with "
                    "the corner handles. It anchors to the detected feature and stays "
                    "put across scans. On non-$1 notes, tick <b>fixed spot</b> if it "
                    "anchors wrongly.")
            self.reset_btn.setVisible(True)
        self._update_crop_info(rect)

    def _has_anchor(self, side, region, key) -> bool:
        """True only if the detector finds a REAL feature to anchor this crop to on
        this sample. We must disable the percentage fallback: otherwise a missing
        detection still returns a (fixed, default) fallback rect, which would look
        like an anchor but ignores the offset/min knobs entirely — exactly what
        made non-$1 seal crops revert to a default."""
        if key is None:
            return False
        tmp = cg.clear_fixed(copy.deepcopy(self._wcfg), key)
        tmp.setdefault('yolo_crops', {})['fallback_on_missing'] = False
        return self._ctx.render(side, region, tmp)[1] is not None

    def _set_default_fixed(self, side, key):
        """Seed a centred default fixed box for a region with no anchor."""
        img = self._ctx.imgs[side]
        h, w = img.shape[:2]
        # a reasonable default: middle third-ish
        x1, y1 = int(w * 0.35), int(h * 0.30)
        x2, y2 = int(w * 0.65), int(h * 0.70)
        cg.set_fixed(self._wcfg, key, (x1, y1, x2, y2), w, h)

    def _update_crop_info(self, rect):
        # Show size AND position -- moving a box keeps its size, so a size-only
        # readout looks frozen while dragging; the @position updates on a move.
        if rect:
            x1, y1, x2, y2 = rect
            self.crop_info.setText(f"crop: {x2 - x1}×{y2 - y1} px   at ({x1}, {y1})")
        else:
            self.crop_info.setText("")

    def _on_crop_drag(self, dragged):
        c = self._cur_crop()
        side, region = c['side'], c['region']
        render = lambda cc: self._ctx.render(side, region, cc)[1]
        if region in cg.THIRDS_REGIONS:
            base = cg.base_thirds_rect(render, self._wcfg, side)
            if base is not None:
                cg.apply_thirds(self._wcfg, side,
                                cg.invert_thirds(base, tuple(dragged), region))
        else:
            key = self._region_key(side, region)
            if key is None:
                return
            h, w = self._ctx.imgs[side].shape[:2]
            if cg.is_fixed(self._wcfg, key):
                cg.set_fixed(self._wcfg, key, tuple(dragged), w, h)
            else:
                base = cg.base_rect(render, self._wcfg, key)
                if base is not None:
                    cg.apply_updates(self._wcfg, key,
                                     cg.invert_drag(base, tuple(dragged), key))
        self._update_crop_info(self._ctx.render(side, region, self._wcfg)[1])

    def _on_enable_toggled(self, on):
        c = self._cur_crop()
        self._enabled[(c['side'], c['region'])] = bool(on)

    def _on_fixed_toggled(self, on):
        """Switch the current box crop between fixed (honour the drawn box) and
        anchored (track the detected feature)."""
        c = self._cur_crop()
        side, region = c['side'], c['region']
        key = self._region_key(side, region)
        if key is None:
            return
        if on:
            # Freeze the current on-screen rect as a fixed box.
            rect = self._ctx.render(side, region, self._wcfg)[1]
            if rect is None:
                self._set_default_fixed(side, key)
            else:
                h, w = self._ctx.imgs[side].shape[:2]
                cg.set_fixed(self._wcfg, key, tuple(rect), w, h)
        else:
            cg.clear_fixed(self._wcfg, key)
        self._enter_crop()

    def _on_reset_box(self):
        c = self._cur_crop()
        side, region = c['side'], c['region']
        if region in cg.THIRDS_REGIONS:
            cg.apply_thirds(self._wcfg, side,
                            {'left_inner': 0, 'right_inner': 0,
                             'center_left': 0, 'center_right': 0})
            self._enter_crop()
            return
        key = self._region_key(side, region)
        if key is None:
            return
        cg.clear_fixed(self._wcfg, key)
        cg.apply_updates(self._wcfg, key,
                         {'offset_x': 0, 'offset_y': 0, 'min_width': 0, 'min_height': 0}
                         if key != 'back_seal'
                         else {'offset_x': 0, 'offset_y': 0})
        self._enter_crop()

    def _finalize_profile(self):
        """Build the profile dict from the working state and stash it."""
        order = [[c['side'], c['region']] for c in _CROPS
                 if self._enabled[(c['side'], c['region'])]]
        denom = _DENOM_NUMBER.get(self._sample['key'])
        profile = {
            'crop_order': order,
            'yolo_crops': copy.deepcopy(self._wcfg.get('yolo_crops', {})),
            'include_serial_overlay': False,
            'min_dimension': 500,
        }
        if denom is not None:
            profile['denomination'] = denom
        # Same name -> OVERWRITE (edit that profile), never pile up as "$20 (2)".
        name = self._profile_name
        self._updated_existing = (name in self.saved_profiles
                                  or name in self._existing)
        self.saved_profiles[name] = profile
        self._last_saved = name

    def _enter_done(self):
        n = len(self.saved_profiles)
        verb = "Updated" if getattr(self, '_updated_existing', False) else "Saved"
        self.done_body.setText(
            f"{verb} the <b>{self._last_saved}</b> profile "
            f"({'1 profile' if n == 1 else f'{n} profiles'} this session).<br><br>"
            "You can <b>reorder</b> the crops, <b>turn any on or off</b>, and "
            "fine-tune the boxes any time from <b>Crop settings…</b> in the main "
            "window.<br><br>"
            "Want to set up another denomination, or move on?"
        )

    def _enter_ready(self):
        names = ", ".join(self.saved_profiles.keys()) or "(none)"
        self.ready_body.setText(
            f"<b>Profiles created:</b> {names}<br>"
            f"<b>Bills folder:</b> {self.input_dir or '(set later)'}<br>"
            f"<b>Crops saved to:</b> {self.output_dir or '(set later)'}<br><br>"
            "Click <b>Finish</b> to load this into the crop tool. Pick a profile, "
            "set your folders if you haven't, and press <b>Run</b> to crop."
        )

    def _browse(self, edit):
        start = edit.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if d:
            edit.setText(d)

    def accept(self):
        # Persist the profiles + choose an active one.
        active = self._last_saved if getattr(self, '_last_saved', None) else None
        if self.saved_profiles and self._save_config:
            self._save_config(self.saved_profiles, active)
        super().accept()
