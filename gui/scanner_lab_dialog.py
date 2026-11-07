"""Tools -> Scanner Test: step-by-step check of what a feed scanner can do.

Walks the user through dollar_detective.scanner.lab.STEPS (find scanner, empty
feeder, front only, both sides, shorter window, speed run), shows a plain
pass/fail checklist, and saves a report .zip they can send back.
"""

import threading
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QSpinBox,
    QVBoxLayout, QWidget)

from dollar_detective.scanner import lab

_ICON = {"pass": "✓", "warn": "!", "fail": "✗", "skip": "–"}


class _Runner(QThread):
    progress = Signal(str)
    done = Signal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn(self.progress.emit))
        except Exception as e:   # never let a scanner hiccup kill the dialog
            r = lab.StepResult(key="?", error=repr(e))
            self.done.emit(r)


class ScannerLabDialog(QDialog):
    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scanner Test")
        self.resize(860, 560)
        self.settings = settings
        self.results = {}
        self.backend = None
        self.device_id = None
        self._runner = None
        self._stop = threading.Event()
        self._index = 0

        root = QHBoxLayout(self)
        self.step_list = QListWidget()
        self.step_list.setFixedWidth(220)
        for s in lab.STEPS:
            self.step_list.addItem(f"   {s.title}")
        self.step_list.addItem("   Results")
        self.step_list.currentRowChanged.connect(self._show_step)
        root.addWidget(self.step_list)

        right = QVBoxLayout()
        root.addLayout(right, 1)
        self.title = QLabel()
        self.title.setStyleSheet("font-size:16px; font-weight:bold;")
        right.addWidget(self.title)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        right.addWidget(self.instructions)

        row = QHBoxLayout()
        self.bills_label = QLabel("Bills loaded:")
        self.bills_spin = QSpinBox()
        self.bills_spin.setRange(1, 200)
        row.addWidget(self.bills_label)
        row.addWidget(self.bills_spin)
        row.addStretch()
        right.addLayout(row)

        self.noise_box = QWidget()
        nl = QHBoxLayout(self.noise_box)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.addWidget(QLabel("Did the rollers move or make noise?"))
        self.noise_yes = QRadioButton("Yes")
        self.noise_no = QRadioButton("No")
        self._noise_group = QButtonGroup(self)
        for b in (self.noise_yes, self.noise_no):
            self._noise_group.addButton(b)
            nl.addWidget(b)
            b.toggled.connect(self._noise_answered)
        nl.addStretch()
        right.addWidget(self.noise_box)

        self.status = QLabel()
        self.status.setWordWrap(True)
        right.addWidget(self.status)

        thumbs = QHBoxLayout()
        self.thumb_a, self.thumb_b = QLabel(), QLabel()
        for t in (self.thumb_a, self.thumb_b):
            t.setMinimumHeight(140)
            t.setAlignment(Qt.AlignCenter)
            thumbs.addWidget(t)
        right.addLayout(thumbs)

        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        right.addWidget(self.report_view, 1)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run_or_stop)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._skip)
        self.next_btn = QPushButton("Next ›")
        self.next_btn.clicked.connect(lambda: self.step_list.setCurrentRow(self._index + 1))
        self.save_btn = QPushButton("Save Report…")
        self.save_btn.clicked.connect(self._save_report)
        self.apply_btn = QPushButton("Use Recommended Settings")
        self.apply_btn.clicked.connect(self._apply)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for b in (self.run_btn, self.skip_btn, self.next_btn, self.save_btn, self.apply_btn):
            btns.addWidget(b)
        btns.addStretch()
        btns.addWidget(close)
        right.addLayout(btns)

        self.step_list.setCurrentRow(0)

    # -- navigation ----------------------------------------------------------
    def _is_results(self, i):
        return i >= len(lab.STEPS)

    def _show_step(self, i):
        if i < 0:
            return
        if self._runner is not None:          # don't wander off mid-scan
            self.step_list.blockSignals(True)
            self.step_list.setCurrentRow(self._index)
            self.step_list.blockSignals(False)
            return
        self._index = i
        results_page = self._is_results(i)
        for w in (self.run_btn, self.skip_btn, self.bills_label, self.bills_spin):
            w.setVisible(not results_page)
        self.next_btn.setVisible(not results_page)
        self.save_btn.setVisible(results_page)
        self.apply_btn.setVisible(results_page and self.settings is not None)
        self.thumb_a.clear()
        self.thumb_b.clear()
        self.thumb_a.setVisible(not results_page)
        self.thumb_b.setVisible(not results_page)
        if results_page:
            self.title.setText("Results")
            self.instructions.setText(
                "What worked and what didn't. Click Save Report… and send the "
                "file to whoever is helping you set up the scanner.")
            self.noise_box.setVisible(False)
            self.status.setText("")
            self._refresh_report()
            return
        step = lab.STEPS[i]
        self.title.setText(step.title)
        self.instructions.setText(step.instructions)
        needs_paper = step.default_bills > 0
        self.bills_label.setVisible(needs_paper)
        self.bills_spin.setVisible(needs_paper)
        if needs_paper:
            prev = self.results.get(step.key)
            self.bills_spin.setValue(prev.bills_loaded if prev else step.default_bills)
        if step.scan is not None and self.backend is None and step.key != "discover":
            self.run_btn.setEnabled(False)
            self.status.setText("Run “Find the scanner” first.")
        else:
            self.run_btn.setEnabled(True)
            self.status.setText(self._describe(self.results.get(step.key)))
        self.run_btn.setText("Run again" if step.key in self.results else "Run")
        self.noise_box.setVisible(step.ask_noise and step.key in self.results)
        self._show_thumbs(self.results.get(step.key))
        self.report_view.setPlainText(self._step_detail(self.results.get(step.key)))

    def _mark(self, key, status):
        i = next(n for n, s in enumerate(lab.STEPS) if s.key == key)
        self.step_list.item(i).setText(f"{_ICON[status]}  {lab.STEPS[i].title}")

    # -- running -------------------------------------------------------------
    def _run_or_stop(self):
        if self._runner is not None:
            self._stop.set()
            self.status.setText("Stopping…")
            return
        step = lab.STEPS[self._index]
        if step.key == "discover":
            self._pick_backend()
            if self.backend is None:
                self.status.setText(
                    "No supported scanner found. Check it's plugged in and on, that "
                    "its own software is closed, then Run again.")
                self.results["discover"] = lab.StepResult(key="discover", error="No scanner found.")
                self._mark("discover", "fail")
                return
        bills = self.bills_spin.value() if step.default_bills else 0
        self._stop.clear()
        backend, dev, results = self.backend, self.device_id, self.results
        dpi = int(getattr(getattr(self.settings, "scanner", None), "dpi", 300) or 300)

        def work(progress):
            return lab.run_step(step, backend, dev, results, bills=bills, dpi=dpi,
                                stop=self._stop, progress=progress)

        self._runner = _Runner(work, self)
        self._runner.progress.connect(self.status.setText)
        self._runner.done.connect(lambda r, k=step.key: self._finished(k, r))
        self.run_btn.setText("Stop")
        self.skip_btn.setEnabled(False)
        self.status.setText("Working…")
        self._runner.start()

    def _finished(self, key, res):
        t = self._runner
        self._runner = None
        if t is not None:
            t.wait(2000)
        res.key = key
        self.results[key] = res
        self.run_btn.setText("Run again")
        self.skip_btn.setEnabled(True)
        self._mark(key, "pass" if res.ok else "fail")
        self._show_step(self._index)
        if key == "empty":
            self.noise_box.setVisible(True)

    def _skip(self):
        step = lab.STEPS[self._index]
        self.results[step.key] = lab.StepResult(key=step.key, skipped=True)
        self._mark(step.key, "skip")
        self.step_list.setCurrentRow(self._index + 1)

    def _noise_answered(self):
        r = self.results.get("empty")
        if r is not None:
            r.noise = "yes" if self.noise_yes.isChecked() else "no"

    def _pick_backend(self):
        from dollar_detective.scanner import available_backends
        self.backend, self.device_id = None, None
        want = getattr(getattr(self.settings, "scanner", None), "device", "auto")
        for b in available_backends():
            try:
                devs = b.list_devices()
            except Exception:
                devs = []
            if devs:
                self.backend = b
                ids = [d.id for d in devs]
                self.device_id = want if want in ids else ids[0]
                return

    # -- display -------------------------------------------------------------
    def _describe(self, r):
        if r is None:
            return ""
        if r.skipped:
            return "Skipped."
        if r.key == "discover":
            return (f"Found: {r.info.get('name') or r.info.get('backend', '')}" if r.ok
                    else f"Not found. {r.error}")
        fronts = sum(1 for p in r.pages if p["side"] != "B")
        backs = len(r.pages) - fronts
        base = f"{len(r.pages)} pages ({fronts} front, {backs} back) in {r.seconds:.1f}s."
        if r.error and not r.ok:
            return f"Problem: {r.error}  {base}"
        return ("Passed. " if r.ok else "") + base

    def _step_detail(self, r):
        if r is None or r.skipped:
            return ""
        import json
        if r.key == "discover":
            return json.dumps(r.info, indent=2, default=str)
        lines = [f"options: {r.options}", ""]
        for i, p in enumerate(r.pages, 1):
            a = p["analysis"]
            if a.get("found"):
                lines.append(
                    f"{i:3d} {p['side']}  t={p['t']:6.2f}s  {a['w']}x{a['h']}px  "
                    f"bill {a['bill_in'][0]}x{a['bill_in'][1]}in  {a['orientation']}  "
                    f"ends {a['bill_end_in']}in  cut={a['trailing_touch']}")
            else:
                lines.append(f"{i:3d} {p['side']}  t={p['t']:6.2f}s  bill NOT found")
        return "\n".join(lines)

    def _show_thumbs(self, r):
        if r is None:
            return
        crops = [v for k, v in sorted(r.samples.items()) if k.endswith("_crop.jpg")]
        for label, blob in zip((self.thumb_a, self.thumb_b), crops):
            pm = QPixmap()
            if pm.loadFromData(blob):
                label.setPixmap(pm.scaled(360, 160, Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation))

    def _refresh_report(self):
        summary = lab.summarize(self.results)
        self.report_view.setPlainText(lab.report_text(self.results, summary))

    # -- results actions -----------------------------------------------------
    def _save_report(self):
        default = str(Path.home() / "Desktop" / "scanner_test_report.zip")
        path, _ = QFileDialog.getSaveFileName(self, "Save Scanner Test Report",
                                              default, "Zip files (*.zip)")
        if not path:
            return
        try:
            from dollar_detective.version import __version__
        except Exception:
            __version__ = "?"
        try:
            lab.save_report(path, self.results, lab.summarize(self.results),
                            extra={"version": __version__})
        except Exception as e:
            QMessageBox.warning(self, "Scanner Test", f"Couldn't save the report:\n{e}")
            return
        QMessageBox.information(self, "Scanner Test", f"Report saved:\n{path}")

    def _apply(self):
        rec = lab.recommend(self.results)
        sc = getattr(self.settings, "scanner", None)
        if sc is None or self.backend is None:
            return
        sc.enabled = True
        sc.duplex = bool(rec.get("duplex"))
        if self.device_id:
            sc.device = self.device_id
        try:
            self.settings.save()
        except Exception:
            pass
        QMessageBox.information(
            self, "Scanner Test",
            "Saved: direct scanning on, "
            f"{'both sides' if sc.duplex else 'front only'}.")

    def closeEvent(self, event):
        if self._runner is not None:
            self._stop.set()
            self._runner.wait(15000)
        super().closeEvent(event)
