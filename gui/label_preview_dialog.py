"""Label Preview & Print dialog — Phase 1 of the label-printing tool.

In-flow WYSIWYG label editor so the FIL can see and fix his labels *in the app*
instead of hand-editing them in Word before printing.

Two layers of control:
- PROFILE (shared by all labels): a named template with its own label SIZE, base
  font, field layout + captions, edited via "Edit…" (LabelFieldsDialog). Change
  "SERIES " to "Ser: " once and every label updates; toggle Catalog/Position on
  for all of them; keep a "2×1" and a "6×4" and switch between them. Persisted to
  settings.label_profiles.
- PER-BILL override: each label's text box lets him tweak one bill's text
  freeform (`label_lines`), label-only; Reset reverts to the profile.

Long lines wrap; the red ⚠ fires only when text runs off the bottom of the
label. Export to PDF (prints as shown) or editable Word .docx.

Later: a printer calibration grid for aligning to the physical label stock.
"""
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QCheckBox,
    QPushButton, QScrollArea, QWidget, QFrame, QFileDialog, QMessageBox,
    QComboBox, QInputDialog,
)

from .label_render import LabelData, LabelTemplate, render_image

_THUMB_W = 280
_THUMB_H = 140
_PREVIEW_DPI = 200.0


class _LabelRow(QFrame):
    """One bill: label thumbnail + editable lines + overflow warning."""

    def __init__(self, data: LabelData, template: LabelTemplate, uppercase: bool,
                 on_lines_changed: Callable[[LabelData, Optional[List[str]]], None],
                 on_overflow_changed: Callable[[], None],
                 on_note_changed: Optional[Callable[[LabelData, str], None]] = None,
                 parent=None):
        super().__init__(parent)
        self.data = data
        self._template = template
        self._uppercase = uppercase
        self._on_lines_changed = on_lines_changed
        self._on_overflow_changed = on_overflow_changed
        self._on_note_changed = on_note_changed
        self.has_overflow = False

        self.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        self.thumb = QLabel()
        self.thumb.setFixedSize(_THUMB_W, _THUMB_H)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("border: 1px solid #999; background: white;")
        layout.addWidget(self.thumb)

        right = QVBoxLayout()
        right.setSpacing(4)
        self.editor = QPlainTextEdit()
        self.editor.setPlainText("\n".join(self._current_lines()))
        self.editor.setFixedHeight(96)
        self.editor.setToolTip("Freeform tweak for THIS bill only (one line per row). "
                               "Reset reverts to the shared template.")
        self.editor.textChanged.connect(self._on_edited)
        right.addWidget(self.editor)

        controls = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Revert this label to the shared template")
        self.reset_btn.setEnabled(self.data.lines is not None)
        self.reset_btn.clicked.connect(self._reset)
        controls.addWidget(self.reset_btn)
        controls.addStretch(1)
        right.addLayout(controls)

        # Inline suggestion hint (only shows when a suggestion is available and the
        # note is still empty) with a one-click "Use".
        self.suggest_row = QWidget()
        sug = QHBoxLayout(self.suggest_row)
        sug.setContentsMargins(0, 0, 0, 0)
        sug.setSpacing(6)
        self.suggest_lbl = QLabel("")
        self.suggest_lbl.setStyleSheet("color: #888;")
        self.suggest_lbl.setWordWrap(True)
        sug.addWidget(self.suggest_lbl, 1)
        self.use_btn = QPushButton("Use")
        self.use_btn.setToolTip("Put this suggestion on the label (line 3)")
        self.use_btn.setFixedWidth(48)
        self.use_btn.clicked.connect(self._use_suggestion)
        sug.addWidget(self.use_btn)
        right.addWidget(self.suggest_row)

        self.warning = QLabel("")
        self.warning.setStyleSheet("color: #c62828;")
        self.warning.setWordWrap(True)
        right.addWidget(self.warning)
        right.addStretch(1)

        layout.addLayout(right, 1)
        self._refresh_thumb()
        self._update_suggestion_hint()

    def _current_lines(self) -> List[str]:
        if self.data.lines is not None:
            return list(self.data.lines)
        return self.data.template_lines(self._template)

    def set_uppercase(self, uppercase: bool):
        self._uppercase = uppercase
        self._refresh_thumb()

    def retemplate(self, template: LabelTemplate):
        """Adopt a new shared template. Bills without an override re-seed from it."""
        self._template = template
        if self.data.lines is None:
            self.editor.blockSignals(True)
            self.editor.setPlainText("\n".join(self.data.template_lines(template)))
            self.editor.blockSignals(False)
        self._refresh_thumb()
        self._update_suggestion_hint()

    def apply_note_value(self, note: str):
        """Set the bill's Note value (a field value, not a freeform override)."""
        self.data.note = note
        if self.data.lines is None:
            self.editor.blockSignals(True)
            self.editor.setPlainText("\n".join(self.data.template_lines(self._template)))
            self.editor.blockSignals(False)
        self._refresh_thumb()
        self._update_suggestion_hint()

    def _update_suggestion_hint(self):
        """Show the suggestion hint only while it's still applicable."""
        show = bool(self.data.suggested_note) and not self.data.note.strip() \
            and self.data.lines is None
        self.suggest_row.setVisible(show)
        if show:
            self.suggest_lbl.setText(f"💡 Suggested line 3:  “{self.data.suggested_note}”")

    def _use_suggestion(self):
        note = self.data.suggested_note
        if not note:
            return
        self.apply_note_value(note)
        if self._on_note_changed:
            self._on_note_changed(self.data, note)
        self._on_overflow_changed()

    def _reset(self):
        self.data.lines = None
        self.editor.blockSignals(True)
        self.editor.setPlainText("\n".join(self.data.template_lines(self._template)))
        self.editor.blockSignals(False)
        self.reset_btn.setEnabled(False)
        self._on_lines_changed(self.data, None)
        self._refresh_thumb()
        self._update_suggestion_hint()
        self._on_overflow_changed()

    def _on_edited(self):
        raw = self.editor.toPlainText()
        lines = raw.split("\n")
        if len(lines) > 1 and lines[-1] == "":
            lines = lines[:-1]
        if lines == self.data.template_lines(self._template):
            self.data.lines = None
            self.reset_btn.setEnabled(False)
            self._on_lines_changed(self.data, None)
        else:
            self.data.lines = lines
            self.reset_btn.setEnabled(True)
            self._on_lines_changed(self.data, lines)
        self._refresh_thumb()
        self._update_suggestion_hint()
        self._on_overflow_changed()

    def _refresh_thumb(self):
        img, overflow = render_image(self.data, self._template, self._uppercase, dpi=_PREVIEW_DPI)
        pix = QPixmap.fromImage(img).scaled(
            _THUMB_W, _THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.thumb.setPixmap(pix)
        self.has_overflow = overflow
        self.warning.setText(
            "⚠ Too many lines to fit — some text is cut off the bottom." if overflow else "")


class LabelPreviewDialog(QDialog):
    """Batch label preview/editor with a shared template and per-bill edits."""

    def __init__(self, items: List[LabelData], settings,
                 on_lines_changed: Optional[Callable[[str, Optional[List[str]]], None]] = None,
                 on_note_changed: Optional[Callable[[str, str], None]] = None,
                 source_desc: str = "", parent=None):
        super().__init__(parent)
        self.settings = settings
        self._items = items
        self._persist = on_lines_changed
        self._persist_note = on_note_changed
        self.profile_name = settings.get_active_label_profile()
        self.template = LabelTemplate.from_dict(settings.get_label_template(self.profile_name))
        self.setWindowTitle("Label Preview & Print")
        self.resize(660, 800)
        self._rows: List[_LabelRow] = []
        self._build_ui(source_desc)
        self._update_summary()

    def _build_ui(self, source_desc: str):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        src = source_desc or f"{len(self._items)} label(s)"
        header.addWidget(QLabel(src))
        header.addStretch(1)
        self.upper_chk = QCheckBox("UPPERCASE")
        self.upper_chk.setChecked(bool(getattr(self.settings.ui, 'label_uppercase', False)))
        self.upper_chk.setToolTip("Force all label text to capitals (house style)")
        self.upper_chk.toggled.connect(self._on_uppercase_toggled)
        header.addWidget(self.upper_chk)
        layout.addLayout(header)

        # Profile row: pick / create / edit the label profile (size + layout).
        prof_row = QHBoxLayout()
        prof_row.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip("Label profile — its size, fields and fonts")
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        prof_row.addWidget(self.profile_combo)
        for text, tip, slot in (
                ("New…", "Create a new profile (e.g. a 6×4 label)", self._new_profile),
                ("Edit…", "Edit this profile's size, fields, captions and fonts", self._edit_profile),
                ("Rename…", "Rename this profile", self._rename_profile),
                ("Delete", "Delete this profile", self._delete_profile)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            prof_row.addWidget(b)
        prof_row.addStretch(1)
        layout.addLayout(prof_row)
        self._reload_profile_combo()

        hint = QLabel("Edit… changes size/captions/layout for every label on this profile. "
                      "The box under each label tweaks that one bill only (Reset reverts).")
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.summary = QLabel("")
        layout.addWidget(self.summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(6)
        uppercase = self.upper_chk.isChecked()
        if not self._items:
            vbox.addWidget(QLabel("No bills to label."))
        for data in self._items:
            row = _LabelRow(data, self.template, uppercase, self._on_row_lines_changed,
                            self._update_summary, self._on_row_note_changed, container)
            self._rows.append(row)
            vbox.addWidget(row)
        vbox.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        btns = QHBoxLayout()
        have_suggestions = any(d.suggested_note for d in self._items)
        self.suggest_btn = QPushButton("Suggest line 3")
        self.suggest_btn.setToolTip("Fill empty notes with an auto-derived feature note "
                                    "(ladder digits, grouping…) from each bill's pattern")
        self.suggest_btn.clicked.connect(self._suggest_notes)
        self.suggest_btn.setEnabled(have_suggestions)
        btns.addWidget(self.suggest_btn)
        btns.addStretch(1)
        self.pdf_btn = QPushButton("Save PDF…")
        self.pdf_btn.setToolTip("Export these labels to a 2x1 inch PDF (prints exactly as shown)")
        self.pdf_btn.clicked.connect(self._save_pdf)
        self.pdf_btn.setEnabled(bool(self._items))
        btns.addWidget(self.pdf_btn)
        self.docx_btn = QPushButton("Save Word…")
        self.docx_btn.setToolTip("Export to an editable Word .docx you can tweak before printing")
        self.docx_btn.clicked.connect(self._save_docx)
        self.docx_btn.setEnabled(bool(self._items))
        btns.addWidget(self.docx_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    def _preview_sample(self) -> LabelData:
        """A representative label for the field editor's live preview.

        Uses the first bill but fills any empty value with a placeholder so every
        field (incl. catalog/position) renders while he tunes captions.
        """
        base = self._items[0] if self._items else LabelData()
        return LabelData(
            serial=base.serial or "A12345678B",
            series=base.series or "2013",
            pattern=base.pattern or "4 Digit Ladder",
            note=base.note or "note",
            catalog=base.catalog or "A1",
            position=str(base.position) if base.position else "7",
            denomination=base.denomination or "$1",
            front_plate=base.front_plate or "A1",
            back_plate=base.back_plate or "B2",
        )

    def _save_settings(self):
        try:
            self.settings.save()
        except Exception:
            pass

    def _reload_profile_combo(self):
        """Repopulate the profile dropdown, selecting the active one."""
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(list(self.settings.get_label_profiles().keys()))
        idx = self.profile_combo.findText(self.profile_name)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def _apply_template(self):
        """Push the current template to every row and refresh the summary."""
        for row in self._rows:
            row.retemplate(self.template)
        self._update_summary()

    def _on_profile_changed(self, name: str):
        if not name or name == self.profile_name:
            return
        self.profile_name = name
        self.settings.set_active_label_profile(name)
        self._save_settings()
        self.template = LabelTemplate.from_dict(self.settings.get_label_template(name))
        self._apply_template()

    def _edit_profile(self):
        from .label_fields_dialog import LabelFieldsDialog
        dlg = LabelFieldsDialog(self.template, self._preview_sample(),
                                self.upper_chk.isChecked(), self.profile_name, self)
        if dlg.exec() and dlg.result_template is not None:
            self.template = dlg.result_template
            self.settings.save_label_profile(self.profile_name, self.template.to_dict())
            self._save_settings()
            self._apply_template()

    def _new_profile(self):
        name, ok = QInputDialog.getText(
            self, "New Label Profile",
            "Name for the new profile (e.g. \"6×4\"):")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in self.settings.get_label_profiles():
            QMessageBox.warning(self, "New Profile", f"A profile named “{name}” already exists.")
            return
        # Seed the new profile from the current one so he starts from something.
        self.settings.save_label_profile(name, self.template.to_dict())
        self.profile_name = name
        self._save_settings()
        self._reload_profile_combo()
        # Jump straight into editing the new profile (size is the usual first change).
        self._edit_profile()

    def _rename_profile(self):
        new, ok = QInputDialog.getText(
            self, "Rename Profile", "New name:", text=self.profile_name)
        new = (new or "").strip()
        if not ok or not new or new == self.profile_name:
            return
        if not self.settings.rename_label_profile(self.profile_name, new):
            QMessageBox.warning(self, "Rename Profile",
                                f"Couldn't rename to “{new}” (name already in use?).")
            return
        self.profile_name = new
        self._save_settings()
        self._reload_profile_combo()

    def _delete_profile(self):
        if len(self.settings.get_label_profiles()) <= 1:
            QMessageBox.information(self, "Delete Profile",
                                    "You need at least one profile.")
            return
        if QMessageBox.question(
                self, "Delete Profile",
                f"Delete the “{self.profile_name}” profile?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.settings.delete_label_profile(self.profile_name)
        self._save_settings()
        self.profile_name = self.settings.get_active_label_profile()
        self.template = LabelTemplate.from_dict(self.settings.get_label_template(self.profile_name))
        self._reload_profile_combo()
        self._apply_template()

    def _on_uppercase_toggled(self, checked: bool):
        self.settings.ui.label_uppercase = checked
        try:
            self.settings.save()
        except Exception:
            pass
        for row in self._rows:
            row.set_uppercase(checked)
        self._update_summary()

    def _on_row_lines_changed(self, data: LabelData, lines: Optional[List[str]]):
        if self._persist and data.front_file:
            self._persist(data.front_file, lines)

    def _on_row_note_changed(self, data: LabelData, note: str):
        if self._persist_note and data.front_file:
            self._persist_note(data.front_file, note)

    def _suggest_notes(self):
        """Fill empty Note values with the auto-derived per-bill suggestion."""
        note_enabled = any(f.var == "note" and f.enabled for f in self.template.fields)
        filled = skipped = 0
        for row in self._rows:
            d = row.data
            if not d.suggested_note:
                continue
            if d.lines is not None or d.note.strip():
                skipped += 1   # keep his own note / freeform override
                continue
            row.apply_note_value(d.suggested_note)
            if self._persist_note and d.front_file:
                self._persist_note(d.front_file, d.suggested_note)
            filled += 1
        self._update_summary()
        msg = f"Filled {filled} note(s) from pattern data."
        if skipped:
            msg += f"\nSkipped {skipped} that already had a note or an edit."
        if filled and not note_enabled:
            msg += ("\n\nNote: the Note field is turned off in this profile, so the "
                    "suggestions won't show until you enable it in Edit….")
        QMessageBox.information(self, "Suggest line 3", msg)

    def _update_summary(self):
        n_over = sum(1 for r in self._rows if r.has_overflow)
        if n_over:
            self.summary.setText(
                f"⚠ {n_over} of {len(self._rows)} label(s) have text that runs off "
                "the label — shorten a line or remove a field.")
            self.summary.setStyleSheet("color: #c62828; font-weight: bold;")
        else:
            self.summary.setText(f"All {len(self._rows)} label(s) fit.")
            self.summary.setStyleSheet("color: #2e7d32;")

    def _save_pdf(self):
        from .label_render import export_pdf
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Labels PDF", "labels.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            n = export_pdf(path, self._items, self.template, self.upper_chk.isChecked())
        except Exception as e:
            QMessageBox.critical(self, "Save PDF", f"Could not save PDF:\n{e}")
            return
        QMessageBox.information(self, "Save PDF", f"Saved {n} label(s) to:\n{path}")

    def _save_docx(self):
        from .label_render import export_docx
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Labels Word Document", "labels.docx", "Word Documents (*.docx)")
        if not path:
            return
        if not path.lower().endswith(".docx"):
            path += ".docx"
        try:
            n = export_docx(path, self._items, self.template, self.upper_chk.isChecked())
        except ImportError:
            QMessageBox.warning(
                self, "Save Word",
                "Word export needs the python-docx package, which isn't installed.")
            return
        except Exception as e:
            QMessageBox.critical(self, "Save Word", f"Could not save the document:\n{e}")
            return
        QMessageBox.information(self, "Save Word", f"Saved {n} label(s) to:\n{path}")
