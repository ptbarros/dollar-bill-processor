"""Help → Send to Paul…: one window to send logs, settings and example bills.

Builds the report zip with dollar_detective.send_report and uploads it on a
worker thread (so the window doesn't freeze), then shows the reference code.
If sending fails, the tester can save the report file instead and email it.
A copy of every report is kept under <user data>/reports/.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QListWidget, QRadioButton, QVBoxLayout, QWidget,
)

from dollar_detective import send_report as sr


class _SendWorker(QThread):
    done = Signal(str)        # reference code
    failed = Signal(str)      # plain-language error

    def __init__(self, build_kwargs: dict, zip_path: Path, url: str, app_version: str,
                 group_code: str = ""):
        super().__init__()
        self.build_kwargs, self.zip_path = build_kwargs, zip_path
        self.url, self.app_version, self.group_code = url, app_version, group_code

    def run(self):
        try:
            sr.build_report(self.zip_path, **self.build_kwargs)
            ref = sr.upload_report(self.zip_path, url=self.url, app_version=self.app_version,
                                   group_code=self.group_code)
            self.done.emit(ref)
        except sr.ReportError as e:
            self.failed.emit(str(e))
        except Exception as e:                       # never crash the app over a report
            self.failed.emit(f"Something went wrong building the report: {e}")


class SendReportDialog(QDialog):
    def __init__(self, parent=None, *, settings=None, current_bill: Optional[dict] = None,
                 queued_bills: Optional[List[dict]] = None, log_paths=(), crash_path=None,
                 settings_path=None, reports_dir: Path = None, app_version: str = "",
                 review_folder: Optional[Path] = None,
                 url: str = sr.REPORT_URL):
        super().__init__(parent)
        self.setWindowTitle("Send to Paul")
        self.settings = settings
        self.current_bill = current_bill
        self.queued = list(queued_bills or [])
        self.log_paths, self.crash_path = list(log_paths), crash_path
        self.settings_path = settings_path
        self.review_folder = Path(review_folder) if review_folder else None
        self._sent_review = False
        self.reports_dir = Path(reports_dir) if reports_dir else Path.home()
        self.app_version, self.url = app_version, url
        self._worker = None
        self._zip_path: Optional[Path] = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Send Paul what he needs to look into a problem or an odd bill. "
            "Folder paths, your account name and API keys are removed first.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        form = QFormLayout()
        self.name_edit = QLineEdit(getattr(getattr(settings, "ui", None), "report_sender_name", "") or "")
        self.name_edit.setPlaceholderText("So Paul knows who sent it")
        form.addRow("Your name:", self.name_edit)
        self.code_edit = QLineEdit(getattr(getattr(settings, "ui", None), "report_group_code", "") or "")
        self.code_edit.setPlaceholderText("From Paul — you only need to enter it once")
        self.code_edit.setToolTip("The group code Paul gave you. It keeps strangers from sending reports.")
        form.addRow("Group code:", self.code_edit)
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText(
            "What happened, or what's odd about the bill(s)? What were you doing at the time?")
        self.note_edit.setMinimumHeight(90)
        form.addRow("What's going on:", self.note_edit)
        lay.addLayout(form)

        lay.addWidget(QLabel("<b>Include</b>"))
        self.logs_cb = QCheckBox("Debug log and computer info (recommended)")
        self.logs_cb.setChecked(True)
        lay.addWidget(self.logs_cb)
        self.settings_cb = QCheckBox("My settings (API keys removed)")
        self.settings_cb.setChecked(True)
        lay.addWidget(self.settings_cb)

        self.bill_group = QButtonGroup(self)
        self.no_bills_rb = QRadioButton("No bills")
        self.current_rb = QRadioButton(self._current_label())
        self.queued_rb = QRadioButton(self._queued_label())
        for i, rb in enumerate((self.no_bills_rb, self.current_rb, self.queued_rb)):
            self.bill_group.addButton(rb, i)
            lay.addWidget(rb)
        self.current_rb.setEnabled(bool(self.current_bill))
        self.queued_rb.setEnabled(bool(self.queued))
        (self.current_rb if self.current_bill else self.no_bills_rb).setChecked(True)
        hint = QLabel("Bills are sent as their front and back scans plus what the app read from them.")
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        n_review = sr.review_count(self.review_folder)
        self.review_cb = QCheckBox(
            f"My saved-for-review bills ({n_review}), with my notes" if n_review
            else "My saved-for-review bills (none saved)")
        self.review_cb.setToolTip(
            "Bills you right-clicked and chose \"Save for Review…\": their scans, "
            "serial close-ups and your note for each.")
        self.review_cb.setEnabled(bool(n_review))
        self.review_cb.setChecked(bool(n_review))
        lay.addWidget(self.review_cb)

        # Screenshots: a one-click picture of the app window, or image files.
        lay.addWidget(QLabel("<b>Screenshots</b> (optional, up to %d)" % sr.MAX_SCREENSHOTS))
        self._shots: List[Path] = []
        self.shot_list = QListWidget()
        self.shot_list.setMaximumHeight(64)
        lay.addWidget(self.shot_list)
        srow = QHBoxLayout()
        self.window_shot_btn = QPushButton("Add a picture of this window")
        self.window_shot_btn.setToolTip("Captures the Dollar Detective main window as it is right now")
        self.window_shot_btn.clicked.connect(self._add_window_shot)
        self.window_shot_btn.setEnabled(parent is not None)
        self.file_shot_btn = QPushButton("Attach screenshot file…")
        self.file_shot_btn.clicked.connect(self._add_file_shot)
        self.remove_shot_btn = QPushButton("Remove")
        self.remove_shot_btn.clicked.connect(self._remove_shot)
        for b in (self.window_shot_btn, self.file_shot_btn, self.remove_shot_btn):
            srow.addWidget(b)
        srow.addStretch()
        lay.addLayout(srow)
        shot_hint = QLabel("Screenshots are sent as they are, so check they don't show anything private.")
        shot_hint.setStyleSheet("color: gray;")
        shot_hint.setWordWrap(True)
        lay.addWidget(shot_hint)
        self._update_shot_buttons()

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)

        row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy reference")
        self.copy_btn.hide()
        self._ref = ""
        self.copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(self._ref))
        self.save_btn = QPushButton("Save report file…")
        self.save_btn.setToolTip("Save the report as a .zip you can email to Paul instead")
        self.save_btn.hide()
        self.save_btn.clicked.connect(self._save_copy)
        row.addWidget(self.copy_btn)
        row.addWidget(self.save_btn)
        row.addStretch()
        self.buttons = QDialogButtonBox()
        self.send_btn = self.buttons.addButton("Send", QDialogButtonBox.AcceptRole)
        self.close_btn = self.buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.send_btn.clicked.connect(self._send)
        self.close_btn.clicked.connect(self.reject)
        row.addWidget(self.buttons)
        lay.addLayout(row)
        self.resize(540, 620)

    # -- labels ------------------------------------------------------------
    def _current_label(self) -> str:
        if not self.current_bill:
            return "The bill on screen (none selected)"
        return f"The bill on screen ({self.current_bill.get('serial') or 'unknown serial'})"

    def _queued_label(self) -> str:
        n = len(self.queued)
        if not n:
            return "My queued (checked) bills (none queued)"
        extra = f", first {sr.MAX_BILLS} only" if n > sr.MAX_BILLS else ""
        return f"My queued (checked) bills ({n}{extra})"

    # -- screenshots ---------------------------------------------------------
    def _update_shot_buttons(self):
        full = len(self._shots) >= sr.MAX_SCREENSHOTS
        self.window_shot_btn.setEnabled(not full and self.parent() is not None)
        self.file_shot_btn.setEnabled(not full)
        self.remove_shot_btn.setEnabled(bool(self._shots))

    def _add_shot(self, path: Path, label: str):
        self._shots.append(path)
        self.shot_list.addItem(label)
        self._update_shot_buttons()

    def _add_window_shot(self):
        win = self.parent()
        if win is None:
            return
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            path = self.reports_dir / f"window-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            if not win.grab().save(str(path), "PNG"):
                raise OSError("couldn't save the picture")
        except Exception as e:
            QMessageBox.warning(self, "Send to Paul", f"Couldn't capture the window:\n{e}")
            return
        self._add_shot(path, f"Picture of the Dollar Detective window ({datetime.now():%H:%M:%S})")

    def _add_file_shot(self):
        start = Path.home() / "Desktop"
        path, _ = QFileDialog.getOpenFileName(
            self, "Attach screenshot", str(start if start.is_dir() else Path.home()),
            "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        p = Path(path)
        if p.stat().st_size > sr.MAX_SCREENSHOT_BYTES:
            QMessageBox.information(
                self, "Send to Paul",
                f"That picture is over {sr.MAX_SCREENSHOT_BYTES // (1024 * 1024)} MB. Please pick a smaller one.")
            return
        self._add_shot(p, p.name)

    def _remove_shot(self):
        row = self.shot_list.currentRow()
        if row < 0:
            row = len(self._shots) - 1
        if 0 <= row < len(self._shots):
            self._shots.pop(row)
            self.shot_list.takeItem(row)
        self._update_shot_buttons()

    def _chosen_bills(self) -> List[dict]:
        if self.current_rb.isChecked() and self.current_bill:
            return [self.current_bill]
        if self.queued_rb.isChecked():
            return self.queued[:sr.MAX_BILLS]
        return []

    # -- send --------------------------------------------------------------
    def _send(self):
        name = self.name_edit.text().strip()
        note = self.note_edit.toPlainText().strip()
        if not name:
            QMessageBox.information(self, "Send to Paul", "Please enter your name.")
            self.name_edit.setFocus()
            return
        code = self.code_edit.text().strip()
        if not code:
            QMessageBox.information(self, "Send to Paul",
                                    "Please enter the group code Paul gave you.")
            self.code_edit.setFocus()
            return
        problem = sr.group_code_problem(code)
        if problem:
            QMessageBox.information(self, "Send to Paul",
                                    problem + "\n\nCheck it against the code Paul gave you.")
            self.code_edit.setFocus()
            return
        self.code_edit.setText(code)      # show it trimmed, as Paul gave it
        if not note:
            QMessageBox.information(self, "Send to Paul",
                                    "Please add a line about what's going on.")
            self.note_edit.setFocus()
            return
        if self.settings is not None:
            try:
                self.settings.ui.report_sender_name = name
                self.settings.ui.report_group_code = code
                self.settings.save()
            except Exception:
                pass
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._zip_path = self.reports_dir / f"report-{stamp}.zip"
        kwargs = dict(
            sender=name, note=note, app_version=self.app_version,
            log_paths=self.log_paths if self.logs_cb.isChecked() else (),
            crash_path=self.crash_path if self.logs_cb.isChecked() else None,
            settings_path=self.settings_path if self.settings_cb.isChecked() else None,
            bills=self._chosen_bills(),
            screenshots=list(self._shots),
            review_folder=self.review_folder if self.review_cb.isChecked() else None,
            system=None if self.logs_cb.isChecked() else {},
        )
        self._set_busy(True)
        self.status.setText("Packing and sending… (bills with images can take a minute)")
        self._worker = _SendWorker(kwargs, self._zip_path, self.url, self.app_version,
                                   group_code=code)
        self._worker.done.connect(self._on_sent)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _set_busy(self, busy: bool):
        for b in (self.window_shot_btn, self.file_shot_btn, self.remove_shot_btn):
            b.setEnabled(not busy)
        if not busy:
            self._update_shot_buttons()
        self.review_cb.setEnabled(not busy and sr.review_count(self.review_folder) > 0)
        for w in (self.send_btn, self.name_edit, self.code_edit, self.note_edit, self.logs_cb,
                  self.settings_cb, self.no_bills_rb, self.current_rb, self.queued_rb):
            w.setEnabled(not busy)
        if not busy:
            self.current_rb.setEnabled(bool(self.current_bill))
            self.queued_rb.setEnabled(bool(self.queued))
        self.close_btn.setEnabled(not busy)

    def _on_sent(self, ref: str):
        self._set_busy(False)
        self.status.setText(
            f"<b>Sent — thank you!</b><br>Reference: <b>{ref}</b><br>"
            "Mention this reference if you write to Paul about it.")
        self.copy_btn.show()
        self.save_btn.hide()
        self._ref = ref
        self.send_btn.setEnabled(False)
        self.close_btn.setText("Close")
        if self.review_cb.isChecked() and self.review_folder:
            self._offer_review_tidy()

    def _offer_review_tidy(self):
        """After a successful send, offer to move the saved-for-review bills into
        the review folder's sent/ archive so they aren't sent twice."""
        n = sr.review_count(self.review_folder)
        if not n:
            return
        if QMessageBox.question(
                self, "Send to Paul",
                f"Your {n} saved-for-review bill(s) went to Paul.\n\n"
                "Move them out of the review list so they aren't sent again? "
                "(They're kept in the review folder's 'sent' archive.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            return
        try:
            from dollar_detective import review_bundle
            review_bundle.archive_review_folder(self.review_folder, mode="sent")
            self.review_cb.setText("My saved-for-review bills (none saved)")
            self.review_cb.setChecked(False)
            self.review_cb.setEnabled(False)
        except Exception as e:
            QMessageBox.warning(self, "Send to Paul",
                                f"Sent, but couldn't tidy the review folder:\n{e}")

    def _on_failed(self, msg: str):
        self._set_busy(False)
        have_zip = bool(self._zip_path and self._zip_path.exists())
        self.status.setText(
            f"<b>Couldn't send.</b> {msg}"
            + ("<br>You can try again, or save the report file and email it to Paul."
               if have_zip else ""))
        self.save_btn.setVisible(have_zip)

    def _save_copy(self):
        if not (self._zip_path and self._zip_path.exists()):
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save report file", str(Path.home() / self._zip_path.name),
            "Zip files (*.zip)")
        if dest:
            try:
                shutil.copy2(self._zip_path, dest)
                self.status.setText(self.status.text() + f"<br>Saved to {dest}")
            except Exception as e:
                QMessageBox.warning(self, "Send to Paul", f"Couldn't save the file:\n{e}")

    def reject(self):
        if self._worker is not None and self._worker.isRunning():
            return          # don't close mid-send (the thread holds the zip)
        super().reject()
