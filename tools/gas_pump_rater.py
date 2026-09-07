#!/usr/bin/env python3
"""Gas Pump Severity Rater -- a small dev/data-collection tool (run from source).

Purpose: build a calibration dataset that maps a human 1-5 "how bad is the gas
pump" judgment onto the machine's measured vertical deviation (px), so the
detector's threshold/severity can be calibrated to real human perception.
This is NOT part of the shipped app -- it never goes through CI or into the
AppImage. Run it locally:

    python3 tools/gas_pump_rater.py

Design (per Paul):
  * BLIND rating: the computed deviation is hidden until you commit a 1-5, then
    revealed (so your judgment doesn't anchor to the machine number).
  * NO digit boxes drawn (they'd bias your read). You rate, then say which digit
    looks off. The only visual helpers are a crosshair overlay and a manual
    rotate nudge.
  * Toggle between the zoomed serial strip and the full bill front.
  * Each rating is written to a CSV immediately; re-running resumes where you
    left off (already-rated files are skipped).

It reuses ProductionProcessor so the machine number matches what the main app
shows: align_for_preview -> YOLO serial boxes (conf 0.3) -> analyze_gas_pump_
digits on the tight crop, taking the max deviation across the (up to 2) regions.
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
    QVBoxLayout, QFileDialog, QMessageBox, QFrame, QSizePolicy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CSV_FIELDS = [
    "timestamp", "filename", "serial", "region_index", "n_regions",
    "human_rating", "human_digit_pos", "machine_max_deviation",
    "machine_worst_digit_pos", "machine_is_gas_pump", "rotation_deg", "notes",
]


# --------------------------------------------------------------------------- #
# Image view: paints a pixmap fit-to-widget with an optional mouse crosshair.
# --------------------------------------------------------------------------- #
class RatingImageView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 320)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: QPixmap | None = None
        self._crosshair = False
        self._mouse = QPoint(-1, -1)
        self.setStyleSheet("background: #202020;")

    def set_pixmap(self, pm: QPixmap | None):
        self._pixmap = pm
        self.update()

    def set_crosshair(self, on: bool):
        self._crosshair = on
        self.update()

    def crosshair_on(self) -> bool:
        return self._crosshair

    def mouseMoveEvent(self, e):
        self._mouse = e.position().toPoint()
        if self._crosshair:
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#202020"))
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        else:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter, "No image")
        if self._crosshair and self._mouse.x() >= 0:
            pen = QPen(QColor(255, 0, 0, 200))
            pen.setWidth(1)
            p.setPen(pen)
            p.drawLine(0, self._mouse.y(), self.width(), self._mouse.y())
            p.drawLine(self._mouse.x(), 0, self._mouse.x(), self.height())
        p.end()


def _cv_to_pixmap(img: np.ndarray) -> QPixmap:
    """BGR/gray ndarray -> QPixmap."""
    if img is None or img.size == 0:
        return QPixmap()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate around center, expanding canvas so nothing is clipped."""
    if abs(deg) < 1e-3 or img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (nw - w) / 2
    m[1, 2] += (nh - h) / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=(32, 32, 32))


class GasPumpRater(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gas Pump Severity Rater")
        self.resize(1100, 760)

        self.processor = None
        self.files: list[Path] = []
        self.idx = 0
        self.csv_path: Path | None = None
        self.rated: set[str] = set()

        # Per-bill computed state
        self._analysis: dict | None = None   # cached analysis for current bill
        self._region = 0                      # which serial region is shown
        self._show_full = False               # zoomed serial vs full bill
        self._rotation = 0.0
        self._pending_rating: int | None = None
        self._pending_digit: int | None = None

        self._build_ui()

    # ---- UI ---------------------------------------------------------------- #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top bar
        top = QHBoxLayout()
        self.folder_btn = QPushButton("Open Folder…")
        self.folder_btn.clicked.connect(self._choose_folder)
        top.addWidget(self.folder_btn)
        self.progress_lbl = QLabel("No folder loaded")
        top.addWidget(self.progress_lbl, 1)
        self.csv_lbl = QLabel("")
        self.csv_lbl.setStyleSheet("color:#888;")
        top.addWidget(self.csv_lbl)
        root.addLayout(top)

        # Image
        self.view = RatingImageView()
        root.addWidget(self.view, 1)

        # Machine reveal (hidden until a rating is committed)
        self.reveal_lbl = QLabel("")
        self.reveal_lbl.setAlignment(Qt.AlignCenter)
        self.reveal_lbl.setStyleSheet(
            "font-size:13px; padding:4px; color:#ddd; background:#333;")
        root.addWidget(self.reveal_lbl)

        # Rating row
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Severity:"))
        self.rating_btns = {}
        labels = {1: "1 none", 2: "2 slight", 3: "3 clear", 4: "4 strong", 5: "5 extreme"}
        for n in range(1, 6):
            b = QPushButton(labels[n])
            b.setCheckable(True)
            b.clicked.connect(lambda _c, v=n: self._set_rating(v))
            rate_row.addWidget(b)
            self.rating_btns[n] = b
        rate_row.addSpacing(20)
        self.skip_btn = QPushButton("Skip (S)")
        self.skip_btn.clicked.connect(self._skip)
        rate_row.addWidget(self.skip_btn)
        root.addLayout(rate_row)

        # Digit position row (populated per bill)
        self.digit_row = QHBoxLayout()
        self.digit_row.addWidget(QLabel("Which digit is off:"))
        self.digit_btns: list[QPushButton] = []
        self.digit_container = QWidget()
        self.digit_container.setLayout(self.digit_row)
        root.addWidget(self.digit_container)

        # Nav row
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev (←)")
        self.prev_btn.clicked.connect(self._prev)
        nav.addWidget(self.prev_btn)
        self.commit_btn = QPushButton("Save & Next (Enter)")
        self.commit_btn.clicked.connect(self._commit_and_next)
        nav.addWidget(self.commit_btn, 1)
        self.next_btn = QPushButton("Skip file (→)")
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)

        # Help
        help_lbl = QLabel(
            "Keys:  1-5 rate  •  click a digit for position (or 0 = none)  •  "
            "F full bill / serial  •  Tab switch region  •  X crosshair  •  "
            "[ ] rotate ∓0.5°  •  \\ reset rotation  •  S skip  •  ← → prev/next")
        help_lbl.setStyleSheet("color:#888; font-size:11px;")
        help_lbl.setWordWrap(True)
        root.addWidget(help_lbl)

    # ---- Folder / processor ------------------------------------------------ #
    def _ensure_processor(self) -> bool:
        if self.processor is not None:
            return True
        try:
            from process_production import ProductionProcessor
            from settings_manager import get_settings
            model = REPO_ROOT / "best.pt"
            if not model.exists():
                QMessageBox.critical(self, "Missing model", f"best.pt not found at {model}")
                return False
            try:
                use_gpu = bool(get_settings().processing.use_gpu)
            except Exception:
                use_gpu = True
            self.setCursor(Qt.WaitCursor)
            self.processor = ProductionProcessor(str(model), use_gpu=use_gpu)
            self.unsetCursor()
            return True
        except Exception as e:  # pragma: no cover - dev tool
            self.unsetCursor()
            QMessageBox.critical(self, "Processor error", f"Could not load processor:\n{e}")
            return False

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder of gas-pump bills")
        if not d:
            return
        if not self._ensure_processor():
            return
        folder = Path(d)
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS and p.is_file())
        # If the folder is organized Dollar_NNN.jpg, fronts are the odd numbers.
        dollar = [(m := re.match(r"Dollar_(\d+)$", p.stem)) and (int(m.group(1)), p)
                  for p in files]
        if all(dollar):
            files = [p for num, p in sorted(dollar) if num % 2 == 1]
        if not files:
            QMessageBox.information(self, "Empty", "No images found in that folder.")
            return
        self.files = files
        self.csv_path = folder / "gas_pump_ratings.csv"
        self._load_existing_csv()
        # Jump to first unrated file.
        self.idx = 0
        for i, p in enumerate(self.files):
            if p.name not in self.rated:
                self.idx = i
                break
        self.csv_lbl.setText(f"→ {self.csv_path.name}")
        self._load_current()

    def _load_existing_csv(self):
        self.rated.clear()
        if self.csv_path and self.csv_path.exists():
            try:
                with open(self.csv_path, newline="") as f:
                    for row in csv.DictReader(f):
                        if row.get("filename"):
                            self.rated.add(row["filename"])
            except Exception:
                pass

    # ---- Analysis ---------------------------------------------------------- #
    def _analyze(self, path: Path) -> dict | None:
        """Reproduce the app's serial-crop + gas-pump path for one front image."""
        proc = self.processor
        aligned, _info = proc.align_for_preview(path)
        img = aligned if aligned is not None else cv2.imread(str(path))
        if img is None:
            return None
        results = proc.yolo_model(img, verbose=False, conf=0.3)
        serial_cls = proc.YOLO_CLASSES.get("serial_number", 7)
        boxes = []
        for r in results:
            for box in getattr(r, "boxes", []) or []:
                if getattr(box, "cls", None) is None:
                    continue
                if int(box.cls[0]) == serial_cls:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        if not boxes:
            return None
        boxes.sort(key=lambda b: (-b[1], b[0]))  # bottom-first, matches app
        boxes = boxes[:2]
        h, w = img.shape[:2]
        regions = []
        for (x1, y1, x2, y2, _c) in boxes:
            tight = img[y1:y2, x1:x2]
            gp = proc.analyze_gas_pump_digits(tight)
            digits = [d for d in gp.get("digit_boxes", []) if not d["is_letter"]]
            worst_pos = None
            if digits:
                worst = max(range(len(digits)), key=lambda i: digits[i]["deviation"])
                if digits[worst]["deviation"] > 0:
                    worst_pos = worst + 1  # 1-based among numeric digits
            # padded crop for display
            pad = 15
            disp = img[max(0, y1 - pad):min(h, y2 + pad),
                       max(0, x1 - pad):min(w, x2 + pad)].copy()
            serial = None
            try:
                s, _conf = proc.extract_serial_from_crop(disp)
                serial = s or None
            except Exception:
                serial = None
            regions.append({
                "deviation": float(gp.get("max_deviation", 0.0)),
                "is_gas_pump": bool(gp.get("is_gas_pump", False)),
                "n_digits": len(digits),
                "worst_pos": worst_pos,
                "display": disp,
                "serial": serial,
            })
        # primary region = the one the machine flags worst (drives the number)
        primary = max(range(len(regions)), key=lambda i: regions[i]["deviation"])
        return {"full": img, "regions": regions, "primary": primary}

    # ---- Rendering --------------------------------------------------------- #
    def _load_current(self):
        self._reset_pending()
        self._rotation = 0.0
        self._show_full = False
        self._analysis = None
        n = len(self.files)
        if not (0 <= self.idx < n):
            return
        path = self.files[self.idx]
        self.progress_lbl.setText(
            f"{self.idx + 1} / {n}   ({len(self.rated)} rated)   —   {path.name}")
        self.setCursor(Qt.WaitCursor)
        try:
            self._analysis = self._analyze(path)
        finally:
            self.unsetCursor()
        if self._analysis is None:
            # No serial detected (likely a back / bad scan). Auto-skip forward.
            self.reveal_lbl.setText("No serial region detected — skipping.")
            self.view.set_pixmap(None)
            self._build_digit_buttons(0, None)
            return
        self._region = self._analysis["primary"]
        self._build_digit_buttons(
            self._analysis["regions"][self._region]["n_digits"],
            self._analysis["regions"][self._region]["serial"])
        self._render()

    def _render(self):
        if self._analysis is None:
            return
        if self._show_full:
            base = self._analysis["full"]
        else:
            base = self._analysis["regions"][self._region]["display"]
        self.view.set_pixmap(_cv_to_pixmap(_rotate(base, self._rotation)))

    def _build_digit_buttons(self, n_digits: int, serial: str | None):
        # Clear old buttons (keep the leading label at index 0).
        for b in self.digit_btns:
            self.digit_row.removeWidget(b)
            b.deleteLater()
        self.digit_btns = []
        # Derive digit characters from the serial if it has the expected shape.
        chars = None
        if serial:
            digs = re.sub(r"[^0-9]", "", serial)
            if len(digs) == 8:
                chars = list(digs)
        count = 8 if not n_digits else n_digits
        none_btn = QPushButton("0 none")
        none_btn.setCheckable(True)
        none_btn.clicked.connect(lambda _c: self._set_digit(0))
        self.digit_row.addWidget(none_btn)
        self.digit_btns.append(none_btn)
        for i in range(1, count + 1):
            label = f"{i}"
            if chars and i <= len(chars):
                label = f"{i}:{chars[i - 1]}"
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, v=i: self._set_digit(v))
            self.digit_row.addWidget(b)
            self.digit_btns.append(b)

    # ---- Rating state ------------------------------------------------------ #
    def _reset_pending(self):
        self._pending_rating = None
        self._pending_digit = None
        for b in self.rating_btns.values():
            b.setChecked(False)
        self.reveal_lbl.setText("")

    def _set_rating(self, v: int):
        self._pending_rating = v
        for n, b in self.rating_btns.items():
            b.setChecked(n == v)
        if v == 1:
            self._set_digit(0)  # "none" is implied for a non-gas-pump bill
        self._reveal()

    def _set_digit(self, v: int):
        self._pending_digit = v
        for i, b in enumerate(self.digit_btns):
            b.setChecked(i == v)  # index 0 == "none"

    def _reveal(self):
        """Show the machine number/worst-digit AFTER a rating (blind-then-reveal)."""
        if self._analysis is None or self._pending_rating is None:
            return
        reg = self._analysis["regions"][self._region]
        wp = reg["worst_pos"]
        self.reveal_lbl.setText(
            f"machine: {reg['deviation']:.2f} px   "
            f"(gas pump: {'yes' if reg['is_gas_pump'] else 'no'})   "
            f"worst digit: {wp if wp else '—'}")

    # ---- Persistence / navigation ----------------------------------------- #
    def _write_row(self, rating, digit, notes=""):
        if self.csv_path is None or self._analysis is None:
            return
        reg = self._analysis["regions"][self._region]
        new_file = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "filename": self.files[self.idx].name,
                "serial": reg.get("serial") or "",
                "region_index": self._region,
                "n_regions": len(self._analysis["regions"]),
                "human_rating": rating,
                "human_digit_pos": "" if digit in (None, 0) else digit,
                "machine_max_deviation": f"{reg['deviation']:.3f}",
                "machine_worst_digit_pos": reg["worst_pos"] or "",
                "machine_is_gas_pump": int(reg["is_gas_pump"]),
                "rotation_deg": f"{self._rotation:.1f}",
                "notes": notes,
            })
        self.rated.add(self.files[self.idx].name)

    def _commit_and_next(self):
        if self._analysis is None:
            self._next()
            return
        if self._pending_rating is None:
            QMessageBox.information(self, "Rate first", "Press 1-5 to rate before saving.")
            return
        digit = self._pending_digit if self._pending_rating >= 2 else 0
        self._write_row(self._pending_rating, digit)
        self._next()

    def _skip(self):
        if self._analysis is not None:
            self._write_row("skip", 0, notes="skipped")
        self._next()

    def _next(self):
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self._load_current()
        else:
            QMessageBox.information(self, "Done", "Reached the last file.")

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_current()

    # ---- Keyboard ---------------------------------------------------------- #
    def keyPressEvent(self, e):
        k = e.key()
        if Qt.Key_1 <= k <= Qt.Key_5:
            self._set_rating(k - Qt.Key_0)
        elif k == Qt.Key_0:
            self._set_digit(0)
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            self._commit_and_next()
        elif k == Qt.Key_Right:
            self._next()
        elif k == Qt.Key_Left:
            self._prev()
        elif k == Qt.Key_S:
            self._skip()
        elif k == Qt.Key_F:
            self._show_full = not self._show_full
            self._render()
        elif k == Qt.Key_Tab:
            if self._analysis and len(self._analysis["regions"]) > 1:
                self._region = (self._region + 1) % len(self._analysis["regions"])
                self._build_digit_buttons(
                    self._analysis["regions"][self._region]["n_digits"],
                    self._analysis["regions"][self._region]["serial"])
                self._reveal()
                self._render()
        elif k == Qt.Key_X:
            self.view.set_crosshair(not self.view.crosshair_on())
        elif k == Qt.Key_BracketLeft:
            self._rotation -= 0.5
            self._render()
        elif k == Qt.Key_BracketRight:
            self._rotation += 0.5
            self._render()
        elif k == Qt.Key_Backslash:
            self._rotation = 0.0
            self._render()
        else:
            super().keyPressEvent(e)


def main():
    app = QApplication(sys.argv)
    QGuiApplication.setApplicationName("Gas Pump Severity Rater")
    w = GasPumpRater()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
