#!/usr/bin/env python3
"""
eBay Crop Wizard — standalone prototype (FIL item #6).

Purpose: prototype the *interaction* for defining eBay crops by dragging boxes on
a sample bill, before touching the real EbayCropDialog / processing pipeline.

Core idea being prototyped: the "hybrid per-crop" model.
  - ANCHORED crop: the box you draw picks up whichever YOLO elements it overlaps
    (seal, plate, serial, year, ...). The real crop is recomputed per-bill from
    those element positions + the padding you framed them with, so it survives
    scan shift. This generalizes _generate_front_seal_crop from the real app.
  - FIXED crop: the box is a plain % region, applied as-is (edge-to-edge style).

A crop drawn over detected elements defaults to ANCHORED; one over empty space
defaults to FIXED. On the canvas the SOLID box is what you drew; the DASHED box
(anchored only) is what it actually produces on the current bill — switch to the
other sample serial and watch the dashed box follow the shifted elements.

Scope: single collection, visual boxing, live preview, export to JSON. Multi-
collection management, the one-off/batch runner, and app integration are out of
scope here (see ~/projects/ebay_crop_wizard_scope.md).

Run:  python tools/crop_wizard_prototype.py
Samples default to ~/Pictures/829/fancy_bills (_02 = front full, _03 = back full).
"""

import sys
import json
import copy
from pathlib import Path
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsPixmapItem,
    QGraphicsSimpleTextItem, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QLineEdit, QRadioButton,
    QButtonGroup, QCheckBox, QComboBox, QSpinBox, QGroupBox, QFileDialog,
    QSplitter, QFrame, QMessageBox, QScrollArea,
)
from PySide6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter, QFont
from PySide6.QtCore import Qt, QRectF, QPointF
from shiboken6 import isValid

# --- Reuse the real app's model location + class map --------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from resource_path import app_base
    DEFAULT_MODEL = app_base() / "best.pt"
except Exception:
    DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "best.pt"

# Mirrors process_production.py YOLO_CLASSES (the 11-class model).
YOLO_CLASSES = {
    'ONE_hashed': 0, 'back_plate': 1, 'bill_back': 2, 'bill_front': 3,
    'denomination': 4, 'front_plate': 5, 'seal_f': 6, 'seal_t': 7,
    'serial_number': 8, 'series_year': 9, 'star_symbol': 10,
}
# Classes that make no sense to anchor to (the whole-bill boxes).
NON_ANCHOR = {'bill_front', 'bill_back'}

CLASS_COLORS = {
    'ONE_hashed': '#8888ff', 'back_plate': '#00c8c8', 'denomination': '#c8a000',
    'front_plate': '#00b0ff', 'seal_f': '#ff7f50', 'seal_t': '#ff00ff',
    'serial_number': '#00d000', 'series_year': '#ffa500', 'star_symbol': '#ff0000',
    'bill_front': '#606060', 'bill_back': '#606060',
}
SAMPLE_DIR = Path.home() / "Pictures" / "829" / "fancy_bills"


# =============================================================================
# YOLO detection (lazy-loaded; ~15 lines lifted from _detect_all_objects)
# =============================================================================
class Detector:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._model = None
        self._cache: dict[str, dict] = {}

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))

    def detect(self, img_bgr: np.ndarray, cache_key: str, conf: float = 0.3) -> dict:
        if cache_key in self._cache:
            return self._cache[cache_key]
        self._ensure()
        results = self._model(img_bgr, verbose=False, conf=conf)
        dets = {name: [] for name in YOLO_CLASSES}
        id_to_name = {v: k for k, v in YOLO_CLASSES.items()}
        for result in results:
            for box in result.boxes:
                if box.cls is None:
                    continue
                cls_id = int(box.cls[0])
                name = id_to_name.get(cls_id)
                if name is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets[name].append((x1, y1, x2, y2, float(box.conf[0])))
        self._cache[cache_key] = dets
        return dets


# =============================================================================
# Crop spec + the crop-computation logic being prototyped
# =============================================================================
@dataclass
class CropSpec:
    name: str
    side: str                       # 'front' | 'back'
    mode: str = 'fixed'             # 'anchored' | 'fixed'
    box: dict = field(default_factory=lambda: {'x': 0.0, 'y': 0.0, 'w': 0.2, 'h': 0.2})
    anchor_classes: list = field(default_factory=list)
    serial_pick: str = 'none'      # none | rightmost | leftmost | highest_conf
    pad: dict = field(default_factory=lambda: {'left': 0.05, 'right': 0.05, 'top': 0.08, 'bottom': 0.08})
    min_width: int = 500
    min_height: int = 500


def _clamp(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return x1, y1, x2, y2


def _pick_serials(dets, serial_pick):
    if not dets:
        return dets
    if serial_pick == 'rightmost':
        return [max(dets, key=lambda b: b[0])]
    if serial_pick == 'leftmost':
        return [min(dets, key=lambda b: b[0])]
    if serial_pick == 'highest_conf':
        return [max(dets, key=lambda b: b[4])]
    return dets


def compute_crop(spec: CropSpec, img_w: int, img_h: int, detections: dict):
    """Return (x1,y1,x2,y2) in pixels for this spec on an image of img_w x img_h.

    Anchored: union of the spec's anchor-class detections + padding + min-size.
    Falls back to the fixed % box when anchored but no anchor elements found.
    """
    def fixed():
        b = spec.box
        return _clamp(b['x'] * img_w, b['y'] * img_h,
                      (b['x'] + b['w']) * img_w, (b['y'] + b['h']) * img_h, img_w, img_h)

    if spec.mode != 'anchored' or not spec.anchor_classes:
        return fixed(), 'fixed'

    boxes = []
    for cls in spec.anchor_classes:
        dets = list(detections.get(cls, []))
        if cls == 'serial_number':
            dets = _pick_serials(dets, spec.serial_pick)
        boxes.extend(dets)

    if not boxes:
        return fixed(), 'fallback'   # anchor elements missing on this bill

    ux1 = min(b[0] for b in boxes); uy1 = min(b[1] for b in boxes)
    ux2 = max(b[2] for b in boxes); uy2 = max(b[3] for b in boxes)
    uw = max(1, ux2 - ux1); uh = max(1, uy2 - uy1)

    x1 = ux1 - spec.pad['left'] * uw
    x2 = ux2 + spec.pad['right'] * uw
    y1 = uy1 - spec.pad['top'] * uh
    y2 = uy2 + spec.pad['bottom'] * uh

    if (x2 - x1) < spec.min_width:
        cx = (x1 + x2) / 2
        x1, x2 = cx - spec.min_width / 2, cx + spec.min_width / 2
    if (y2 - y1) < spec.min_height:
        cy = (y1 + y2) / 2
        y1, y2 = cy - spec.min_height / 2, cy + spec.min_height / 2

    return _clamp(x1, y1, x2, y2, img_w, img_h), 'anchored'


def derive_from_box(box_px, detections, img_w, img_h):
    """From a drawn box (x1,y1,x2,y2 px), infer anchored params.

    Returns dict(anchor_classes, serial_pick, pad) or None if the box overlaps
    no anchorable element (caller then makes it a FIXED crop).
    """
    bx1, by1, bx2, by2 = box_px
    selected = {}
    for cls, dets in detections.items():
        if cls in NON_ANCHOR:
            continue
        chosen = []
        for d in dets:
            ix1, iy1 = max(bx1, d[0]), max(by1, d[1])
            ix2, iy2 = min(bx2, d[2]), min(by2, d[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = (d[2] - d[0]) * (d[3] - d[1])
            if area > 0 and inter / area >= 0.5:      # >=50% of the element inside
                chosen.append(d)
        if chosen:
            selected[cls] = chosen
    if not selected:
        return None

    serial_pick = 'rightmost' if 'serial_number' in selected else 'none'
    boxes = []
    for cls, ch in selected.items():
        if cls == 'serial_number':
            ch = _pick_serials(ch, serial_pick)
        boxes.extend(ch)
    ux1 = min(b[0] for b in boxes); uy1 = min(b[1] for b in boxes)
    ux2 = max(b[2] for b in boxes); uy2 = max(b[3] for b in boxes)
    uw = max(1, ux2 - ux1); uh = max(1, uy2 - uy1)
    pad = {
        'left': max(0.0, (ux1 - bx1) / uw), 'right': max(0.0, (bx2 - ux2) / uw),
        'top': max(0.0, (uy1 - by1) / uh), 'bottom': max(0.0, (by2 - uy2) / uh),
    }
    return {'anchor_classes': list(selected), 'serial_pick': serial_pick, 'pad': pad}


# =============================================================================
# Canvas: resizable crop box with 8 handles
# =============================================================================
HANDLE = 9  # handle size in scene px


class CropRectItem(QGraphicsRectItem):
    """A draggable/resizable rectangle (in scene==image pixel coords).

    on_geometry_change(item) is called after a drag/resize completes.
    on_select(item) is called when the body is pressed.
    """
    def __init__(self, rect: QRectF, color: str):
        super().__init__()
        self.setPos(0, 0)
        self.setRect(rect)
        self.setPen(QPen(QColor(color), 2))
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self.on_geometry_change = None
        self.on_select = None
        self._drag_mode = None       # 'move' or a handle role
        self._start_scene = None
        self._start_rect = None
        self._handles = {}
        for role in ('tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r'):
            h = QGraphicsRectItem(self)
            h.setBrush(QBrush(QColor(color)))
            h.setPen(QPen(QColor('white'), 1))
            h.setZValue(11)
            h.setData(0, role)
            self._handles[role] = h
        self._place_handles()

    def set_color(self, color: str):
        self.setPen(QPen(QColor(color), 2))
        for h in self._handles.values():
            h.setBrush(QBrush(QColor(color)))

    def _place_handles(self):
        r = self.rect()
        s = HANDLE
        pts = {
            'tl': (r.left(), r.top()), 'tr': (r.right(), r.top()),
            'bl': (r.left(), r.bottom()), 'br': (r.right(), r.bottom()),
            't': (r.center().x(), r.top()), 'b': (r.center().x(), r.bottom()),
            'l': (r.left(), r.center().y()), 'r': (r.right(), r.center().y()),
        }
        for role, (x, y) in pts.items():
            self._handles[role].setRect(x - s / 2, y - s / 2, s, s)

    def _handle_at(self, scene_pos):
        for role, h in self._handles.items():
            if h.rect().contains(h.mapFromScene(scene_pos)):
                return role
        return None

    def mousePressEvent(self, ev):
        self._start_scene = ev.scenePos()
        self._start_rect = QRectF(self.rect())
        self._drag_mode = self._handle_at(ev.scenePos()) or 'move'
        if self.on_select:
            self.on_select(self)
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag_mode is None:
            return
        dx = ev.scenePos().x() - self._start_scene.x()
        dy = ev.scenePos().y() - self._start_scene.y()
        r = QRectF(self._start_rect)
        m = self._drag_mode
        if m == 'move':
            r.translate(dx, dy)
        else:
            if 'l' in m:
                r.setLeft(r.left() + dx)
            if 'r' in m:
                r.setRight(r.right() + dx)
            if 't' in m:
                r.setTop(r.top() + dy)
            if 'b' in m:
                r.setBottom(r.bottom() + dy)
        self.setRect(r.normalized())
        self._place_handles()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self._drag_mode = None
        if self.on_geometry_change:
            self.on_geometry_change(self)
        ev.accept()


class CanvasView(QGraphicsView):
    """Shows the current bill, YOLO element overlays, and crop rects.

    Click-drag on empty canvas (in Add mode) rubber-bands a new crop box.
    """
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self._pixmap_item = None
        self._overlay_items = []
        self._faint_items = []
        self._adding = False
        self._rubber_origin = None
        self._rubber_item = None

    def set_add_mode(self, on: bool):
        self._adding = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def load_image(self, img_bgr: np.ndarray):
        sc = self.scene()
        sc.clear()
        self._overlay_items = []
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self._pixmap_item.setZValue(0)
        sc.addItem(self._pixmap_item)
        sc.setSceneRect(0, 0, w, h)
        self.fit()

    def fit(self):
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fit()

    def show_overlays(self, detections: dict, visible: bool):
        for it in self._overlay_items:
            self.scene().removeItem(it)
        self._overlay_items = []
        if not visible or not detections:
            return
        for cls, dets in detections.items():
            color = QColor(CLASS_COLORS.get(cls, '#ffffff'))
            for d in dets:
                x1, y1, x2, y2 = d[:4]
                rect = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
                pen = QPen(color, 1.5)
                pen.setStyle(Qt.DotLine)
                rect.setPen(pen)
                rect.setBrush(QBrush(QColor(0, 0, 0, 0)))
                rect.setZValue(1)
                rect.setData(0, 'overlay')
                self.scene().addItem(rect)
                self._overlay_items.append(rect)
                label = QGraphicsSimpleTextItem(cls)
                label.setBrush(QBrush(color))
                f = QFont(); f.setPointSize(8); label.setFont(f)
                label.setPos(x1, max(0, y1 - 14))
                label.setZValue(1)
                self.scene().addItem(label)
                self._overlay_items.append(label)

    def show_faint(self, rects):
        """Draw non-interactive faint outlines (list of (x1,y1,x2,y2,label))."""
        self.clear_faint()
        for x1, y1, x2, y2, label in rects:
            it = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
            pen = QPen(QColor(180, 180, 180, 140), 1.5)
            it.setPen(pen); it.setBrush(QBrush(QColor(0, 0, 0, 0)))
            it.setZValue(2)
            self.scene().addItem(it)
            self._faint_items.append(it)
            if label:
                t = QGraphicsSimpleTextItem(label)
                t.setBrush(QBrush(QColor(160, 160, 160)))
                f = QFont(); f.setPointSize(8); t.setFont(f)
                t.setPos(x1 + 2, y1 + 2); t.setZValue(2)
                self.scene().addItem(t)
                self._faint_items.append(t)

    def clear_faint(self):
        for it in self._faint_items:
            if isValid(it) and it.scene() is self.scene():
                self.scene().removeItem(it)
        self._faint_items = []

    def add_scene_item(self, item):
        self.scene().addItem(item)

    def remove_scene_item(self, item):
        # scene.clear() may already have deleted the underlying C++ object
        # (e.g. after load_image); guard against the stale Python wrapper.
        if item is not None and isValid(item) and item.scene() is self.scene():
            self.scene().removeItem(item)

    # --- rubber-band creation of a new crop -------------------------------
    def mousePressEvent(self, ev):
        if self._adding and ev.button() == Qt.LeftButton:
            self._rubber_origin = self.mapToScene(ev.pos())
            self._rubber_item = QGraphicsRectItem(QRectF(self._rubber_origin, self._rubber_origin))
            pen = QPen(QColor('#ffffff'), 2); pen.setStyle(Qt.DashLine)
            self._rubber_item.setPen(pen)
            self._rubber_item.setZValue(20)
            self.scene().addItem(self._rubber_item)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._rubber_item is not None:
            cur = self.mapToScene(ev.pos())
            self._rubber_item.setRect(QRectF(self._rubber_origin, cur).normalized())
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._rubber_item is not None:
            r = self._rubber_item.rect().normalized()
            self.scene().removeItem(self._rubber_item)
            self._rubber_item = None
            self.set_add_mode(False)
            if r.width() > 8 and r.height() > 8:
                self.main.create_crop_from_rect(r)
            else:
                self.main.on_draw_cancelled()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


# =============================================================================
# Reusable per-crop editor (used on both the Define step and the Review page)
# =============================================================================
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QStackedWidget, QProgressDialog,
)


class CropEditorWidget(QWidget):
    """Edits one CropSpec against the current image/detections, live.

    size_fn() -> (w, h); det_fn() -> detections dict; on_change(spec) is called
    after any edit so the owner can refresh canvas items / thumbnail strips.
    """
    def __init__(self, size_fn, det_fn, on_change, compact=False):
        super().__init__()
        self.size_fn = size_fn
        self.det_fn = det_fn
        self.on_change = on_change
        self.spec: CropSpec | None = None
        self._loading = False
        g = QGridLayout(self)
        r = 0
        g.addWidget(QLabel("Name:"), r, 0)
        self.name_edit = QLineEdit(); self.name_edit.editingFinished.connect(self._apply_name)
        g.addWidget(self.name_edit, r, 1, 1, 2); r += 1

        g.addWidget(QLabel("Type:"), r, 0)
        self.mode_anchored = QRadioButton("Anchored (shift-safe)")
        self.mode_fixed = QRadioButton("Fixed (edge-to-edge)")
        bg = QButtonGroup(self); bg.addButton(self.mode_anchored); bg.addButton(self.mode_fixed)
        self.mode_anchored.toggled.connect(self._apply_mode)
        g.addWidget(self.mode_anchored, r, 1, 1, 2); r += 1
        g.addWidget(self.mode_fixed, r, 1, 1, 2); r += 1

        self.anchor_box = QGroupBox("Anchor to elements")
        ag = QVBoxLayout(self.anchor_box)
        self.class_checks: dict[str, QCheckBox] = {}
        cg = QGridLayout()
        anchorable = [c for c in YOLO_CLASSES if c not in NON_ANCHOR]
        for i, cls in enumerate(anchorable):
            cb = QCheckBox(cls); cb.toggled.connect(self._apply_classes)
            self.class_checks[cls] = cb
            cg.addWidget(cb, i // 2, i % 2)
        ag.addLayout(cg)
        sp = QHBoxLayout(); sp.addWidget(QLabel("2 serials:"))
        self.serial_combo = QComboBox()
        self.serial_combo.addItems(['none', 'rightmost', 'leftmost', 'highest_conf'])
        self.serial_combo.currentTextChanged.connect(self._apply_serial)
        sp.addWidget(self.serial_combo); ag.addLayout(sp)
        mn = QHBoxLayout()
        mn.addWidget(QLabel("Min W:"))
        self.minw = QSpinBox(); self.minw.setRange(0, 4000); self.minw.setValue(500)
        self.minw.valueChanged.connect(self._apply_min); mn.addWidget(self.minw)
        mn.addWidget(QLabel("H:"))
        self.minh = QSpinBox(); self.minh.setRange(0, 4000); self.minh.setValue(500)
        self.minh.valueChanged.connect(self._apply_min); mn.addWidget(self.minh)
        ag.addLayout(mn)
        self.pad_label = QLabel("pad: —"); self.pad_label.setStyleSheet("color:#888")
        ag.addWidget(self.pad_label)
        g.addWidget(self.anchor_box, r, 0, 1, 3); r += 1

        g.addWidget(QLabel("Result on this bill:"), r, 0, 1, 3); r += 1
        self.thumb = QLabel(); self.thumb.setMinimumHeight(130); self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setFrameShape(QFrame.Box)
        g.addWidget(self.thumb, r, 0, 1, 3); r += 1
        self.thumb_info = QLabel(""); self.thumb_info.setStyleSheet("color:#888")
        g.addWidget(self.thumb_info, r, 0, 1, 3)
        self.setEnabled(False)

    def load(self, spec: CropSpec):
        self.spec = spec
        self.setEnabled(True)
        self._loading = True
        self.name_edit.setText(spec.name)
        self.mode_anchored.setChecked(spec.mode == 'anchored')
        self.mode_fixed.setChecked(spec.mode == 'fixed')
        self.anchor_box.setVisible(spec.mode == 'anchored')
        for cls, cb in self.class_checks.items():
            cb.setChecked(cls in spec.anchor_classes)
        self.serial_combo.setCurrentText(spec.serial_pick)
        self.minw.setValue(spec.min_width); self.minh.setValue(spec.min_height)
        self.pad_label.setText(
            "pad L/R/T/B: {left:.2f}/{right:.2f}/{top:.2f}/{bottom:.2f}".format(**spec.pad))
        self._loading = False
        self.refresh_thumb()

    def clear(self):
        self.spec = None
        self.setEnabled(False)
        self.thumb.clear(); self.thumb_info.clear()

    def refresh_thumb(self):
        if not self.spec:
            return
        w, h = self.size_fn()
        (x1, y1, x2, y2), kind = compute_crop(self.spec, w, h, self.det_fn())
        img = self._img()
        if img is None:
            return
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            self.thumb.setText("(empty)"); return
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(rgb.data, cw, ch, 3 * cw, QImage.Format_RGB888).copy()
        self.thumb.setPixmap(QPixmap.fromImage(qimg).scaled(
            max(120, self.thumb.width()), 130, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.thumb_info.setText(f"{cw}×{ch}px  ({kind})")

    # the owner injects the current image getter
    _img_fn = None
    def _img(self):
        return self._img_fn() if self._img_fn else None

    def _changed(self):
        self.refresh_thumb()
        if self.on_change and self.spec:
            self.on_change(self.spec)

    def _apply_name(self):
        if self.spec and not self._loading:
            self.spec.name = self.name_edit.text().strip() or self.spec.name
            self._changed()

    def _apply_mode(self):
        if self._loading or not self.spec:
            return
        self.spec.mode = 'anchored' if self.mode_anchored.isChecked() else 'fixed'
        self.anchor_box.setVisible(self.spec.mode == 'anchored')
        self._changed()

    def _apply_classes(self):
        if self._loading or not self.spec:
            return
        self.spec.anchor_classes = [c for c, cb in self.class_checks.items() if cb.isChecked()]
        self._changed()

    def _apply_serial(self, v):
        if self._loading or not self.spec:
            return
        self.spec.serial_pick = v; self._changed()

    def _apply_min(self):
        if self._loading or not self.spec:
            return
        self.spec.min_width = self.minw.value(); self.spec.min_height = self.minh.value()
        self._changed()


# =============================================================================
# Batch runner: crop every bill in a folder with the current collection
# =============================================================================
class BatchDialog(QDialog):
    def __init__(self, specs, detector, parent=None):
        super().__init__(parent)
        self.specs = specs
        self.detector = detector
        self.setWindowTitle("Batch-crop a folder")
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Crop every bill image in a folder using the crops you defined,\n"
            "independent of the normal processing pipeline. Side (front/back) is\n"
            "auto-detected per image; assumes aligned full-bill images."))
        self.in_edit = self._folder_row(v, "Input folder:")
        self.out_edit = self._folder_row(v, "Output folder:")
        self.log = QLabel(""); self.log.setStyleSheet("color:#888")
        v.addWidget(self.log)
        bb = QDialogButtonBox()
        self.run_btn = bb.addButton("Run", QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Close)
        self.run_btn.clicked.connect(self._run)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _folder_row(self, v, label):
        row = QHBoxLayout(); row.addWidget(QLabel(label))
        edit = QLineEdit(); row.addWidget(edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse(edit))
        row.addWidget(btn); v.addLayout(row)
        return edit

    def _browse(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text() or str(Path.home()))
        if d:
            edit.setText(d)

    def _side_of(self, dets):
        fconf = sum(b[4] for b in dets.get('bill_front', []))
        bconf = sum(b[4] for b in dets.get('bill_back', []))
        return 'back' if bconf > fconf else 'front'

    def _run(self):
        if not self.specs:
            QMessageBox.warning(self, "No crops", "Define at least one crop first.")
            return
        indir = Path(self.in_edit.text()); outdir = Path(self.out_edit.text())
        if not indir.is_dir():
            QMessageBox.warning(self, "Input", "Pick a valid input folder."); return
        if not str(outdir):
            QMessageBox.warning(self, "Output", "Pick an output folder."); return
        outdir.mkdir(parents=True, exist_ok=True)
        imgs = sorted([p for p in indir.iterdir()
                       if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if not imgs:
            QMessageBox.warning(self, "Input", "No images found."); return
        prog = QProgressDialog("Cropping…", "Cancel", 0, len(imgs), self)
        prog.setWindowModality(Qt.WindowModal)
        written = 0
        for i, p in enumerate(imgs):
            prog.setValue(i)
            if prog.wasCanceled():
                break
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            dets = self.detector.detect(img, str(p)) if self.detector.available else {}
            side = self._side_of(dets)
            side_specs = [s for s in self.specs if s.side == side]
            for n, spec in enumerate(side_specs, 1):
                (x1, y1, x2, y2), _ = compute_crop(spec, w, h, dets)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                cv2.imwrite(str(outdir / f"{p.stem}_{n:02d}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                written += 1
            self.log.setText(f"{i+1}/{len(imgs)} — {written} crops written")
            QApplication.processEvents()
        prog.setValue(len(imgs))
        QMessageBox.information(self, "Done", f"Wrote {written} crops to\n{outdir}")


# =============================================================================
# Wizard controller: Start -> Define (step by step) -> Review-all
# =============================================================================
PAGE_START, PAGE_DEFINE, PAGE_REVIEW = 0, 1, 2


class WizardController(QMainWindow):
    def __init__(self, detector: Detector):
        super().__init__()
        self.setWindowTitle("eBay Crop Wizard — Prototype")
        self.resize(1400, 860)
        self.detector = detector
        self.specs: list[CropSpec] = []
        self.samples = self._discover_samples()
        self.current_sample = next(iter(self.samples), None)
        self.current_side = 'front'
        self.img_bgr = None
        self.detections = {}
        self.mode = 'draw'                 # 'draw' (define) | 'edit' (review)
        self.pending_spec: CropSpec | None = None
        self.pending_item: CropRectItem | None = None
        self.crop_items: dict[int, CropRectItem] = {}     # review page (id->solid)
        self.result_items: dict[int, QGraphicsRectItem] = {}
        self.selected_spec: CropSpec | None = None

        self.canvas = CanvasView(self)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_start())
        self.stack.addWidget(self._build_define())
        self.stack.addWidget(self._build_review())
        self.setCentralWidget(self.stack)
        self.go_start()

    # --- samples ----------------------------------------------------------
    def _discover_samples(self) -> dict:
        samples = {}
        if SAMPLE_DIR.exists():
            for f in sorted(SAMPLE_DIR.glob("*_02.jpg")):
                serial = f.name[:-7]
                back = f.with_name(f"{serial}_03.jpg")
                samples[serial] = {'front': str(f),
                                   'back': str(back) if back.exists() else str(f)}
        return samples

    def _cur_size(self):
        if self.img_bgr is None:
            return (1, 1)
        h, w = self.img_bgr.shape[:2]
        return (w, h)

    # =====================================================================
    # PAGE: START
    # =====================================================================
    def _build_start(self):
        page = QWidget(); v = QVBoxLayout(page)
        v.addStretch()
        title = QLabel("eBay Crop Wizard"); title.setAlignment(Qt.AlignCenter)
        f = QFont(); f.setPointSize(20); f.setBold(True); title.setFont(f)
        v.addWidget(title)
        sub = QLabel("Define the crops you want for each bill by drawing them on a sample,\n"
                     "one at a time. You can fine-tune everything at the end.")
        sub.setAlignment(Qt.AlignCenter); sub.setStyleSheet("color:#888")
        v.addWidget(sub)
        v.addSpacing(20)

        row = QHBoxLayout(); row.addStretch()
        row.addWidget(QLabel("Sample bill:"))
        self.start_bill = QComboBox()
        for s in self.samples:
            self.start_bill.addItem(s)
        row.addWidget(self.start_bill)
        row.addWidget(QLabel("  Start side:"))
        self.start_side = QComboBox(); self.start_side.addItems(['front', 'back'])
        row.addWidget(self.start_side)
        row.addStretch(); v.addLayout(row)
        v.addSpacing(16)

        brow = QHBoxLayout(); brow.addStretch()
        begin = QPushButton("Begin defining crops  →"); begin.setMinimumHeight(40)
        begin.clicked.connect(self._on_begin)
        brow.addWidget(begin)
        batch = QPushButton("Batch-crop a folder…"); batch.setMinimumHeight(40)
        batch.clicked.connect(self._open_batch)
        brow.addWidget(batch)
        brow.addStretch(); v.addLayout(brow)
        v.addStretch()
        if not self.samples:
            note = QLabel("No samples found in ~/Pictures/829/fancy_bills — use Load in the app.")
            note.setAlignment(Qt.AlignCenter); note.setStyleSheet("color:#c33")
            v.addWidget(note)
        return page

    def _on_begin(self):
        self.current_sample = self.start_bill.currentText() or self.current_sample
        self.current_side = self.start_side.currentText()
        self.specs.clear()
        self.go_define(fresh=True)

    # =====================================================================
    # PAGE: DEFINE (one crop at a time)
    # =====================================================================
    def _build_define(self):
        page = QWidget(); h = QHBoxLayout(page)
        left = QVBoxLayout()
        self.define_banner = QLabel(""); 
        bf = QFont(); bf.setPointSize(13); bf.setBold(True); self.define_banner.setFont(bf)
        self.define_banner.setStyleSheet("padding:6px; background:#20304a; color:#dbe6ff; border-radius:4px")
        left.addWidget(self.define_banner)
        # side switch + overlay toggle
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Side:"))
        self.define_side = QComboBox(); self.define_side.addItems(['front', 'back'])
        self.define_side.currentTextChanged.connect(self._on_define_side)
        ctl.addWidget(self.define_side)
        self.define_overlay = QCheckBox("Show detected elements"); self.define_overlay.setChecked(True)
        self.define_overlay.toggled.connect(lambda _: self._refresh_overlays())
        ctl.addWidget(self.define_overlay); ctl.addStretch()
        self.define_status = QLabel(""); self.define_status.setStyleSheet("color:#888")
        ctl.addWidget(self.define_status)
        left.addLayout(ctl)
        self.define_canvas_holder = QVBoxLayout()
        left.addLayout(self.define_canvas_holder, 1)
        # accumulated strip
        left.addWidget(QLabel("Crops so far (this side):"))
        self.strip = QScrollArea(); self.strip.setWidgetResizable(True)
        self.strip.setFixedHeight(120)
        self.strip_inner = QWidget(); self.strip_layout = QHBoxLayout(self.strip_inner)
        self.strip_layout.addStretch()
        self.strip.setWidget(self.strip_inner)
        left.addWidget(self.strip)
        h.addLayout(left, 1)

        # right: confirm panel (hidden until a box is drawn)
        self.define_panel = QWidget(); self.define_panel.setFixedWidth(360)
        pv = QVBoxLayout(self.define_panel)
        self.define_editor = CropEditorWidget(self._cur_size, lambda: self.detections,
                                              self._on_pending_changed)
        self.define_editor._img_fn = lambda: self.img_bgr
        pv.addWidget(self.define_editor)
        brow = QHBoxLayout()
        self.redraw_btn = QPushButton("↶ Redraw"); self.redraw_btn.clicked.connect(self._redraw_pending)
        brow.addWidget(self.redraw_btn)
        pv.addLayout(brow)
        arow = QHBoxLayout()
        self.add_next_btn = QPushButton("Add & draw next"); self.add_next_btn.clicked.connect(self._accept_next)
        self.add_finish_btn = QPushButton("Add & finish →"); self.add_finish_btn.clicked.connect(self._accept_finish)
        arow.addWidget(self.add_next_btn); arow.addWidget(self.add_finish_btn)
        pv.addLayout(arow)
        pv.addStretch()
        nrow = QHBoxLayout()
        finish = QPushButton("Finish (skip) →"); finish.clicked.connect(self._finish_define)
        nrow.addStretch(); nrow.addWidget(finish)
        pv.addLayout(nrow)
        h.addWidget(self.define_panel)
        return page

    def go_define(self, fresh=False):
        self.mode = 'draw'
        self.stack.setCurrentIndex(PAGE_DEFINE)
        self.define_side.blockSignals(True)
        self.define_side.setCurrentText(self.current_side)
        self.define_side.blockSignals(False)
        self.define_canvas_holder.addWidget(self.canvas)   # reparent shared canvas
        self._load_image()
        self._clear_pending()
        self._arm_next_draw()
        self._rebuild_strip()

    def _arm_next_draw(self):
        n = len([s for s in self.specs if s.side == self.current_side]) + 1
        self.define_banner.setText(f"Crop #{n} — click and drag a box on the bill")
        self.define_panel.setVisible(False)
        self._show_accepted_faint()
        self.canvas.set_add_mode(True)

    def _on_define_side(self, side):
        self.current_side = side
        self._clear_pending()
        self._load_image()
        self._arm_next_draw()
        self._rebuild_strip()

    # pending-crop lifecycle -------------------------------------------
    def create_crop_from_rect(self, rect: QRectF):
        if self.mode == 'edit':
            self._add_review_crop(rect); return
        w, h = self._cur_size()
        covers = (rect.width() * rect.height()) >= 0.85 * (w * h)
        box_px = (rect.left(), rect.top(), rect.right(), rect.bottom())
        derived = None if covers else derive_from_box(box_px, self.detections, w, h)
        n = len([s for s in self.specs if s.side == self.current_side]) + 1
        spec = CropSpec(name=f"{self.current_side}_{n}", side=self.current_side,
                        box={'x': rect.left()/w, 'y': rect.top()/h,
                             'w': rect.width()/w, 'h': rect.height()/h})
        if derived:
            spec.mode = 'anchored'; spec.anchor_classes = derived['anchor_classes']
            spec.serial_pick = derived['serial_pick']; spec.pad = derived['pad']
        self.pending_spec = spec
        item = CropRectItem(rect, '#00d000' if spec.mode == 'anchored' else '#00b0ff')
        item._spec = spec
        item.on_geometry_change = self._on_pending_geometry
        item.on_select = lambda it: None
        self.canvas.add_scene_item(item)
        self.pending_item = item
        self.define_banner.setText(f"Crop #{n} — name it and confirm, or Redraw")
        self.define_panel.setVisible(True)
        self.define_editor.load(spec)
        self._update_pending_result()

    def on_draw_cancelled(self):
        # too-small drag: just re-arm
        if self.pending_spec is None:
            self._arm_next_draw()

    def _on_pending_geometry(self, item):
        spec = item._spec; w, h = self._cur_size(); r = item.rect()
        spec.box = {'x': r.left()/w, 'y': r.top()/h, 'w': r.width()/w, 'h': r.height()/h}
        if spec.mode == 'anchored':
            d = derive_from_box((r.left(), r.top(), r.right(), r.bottom()), self.detections, w, h)
            if d:
                spec.anchor_classes = d['anchor_classes']
                spec.serial_pick = d['serial_pick']; spec.pad = d['pad']
        self.define_editor.load(spec)
        self._update_pending_result()

    def _on_pending_changed(self, spec):
        if self.pending_item is not None:
            self.pending_item.set_color('#00d000' if spec.mode == 'anchored' else '#00b0ff')
        self._update_pending_result()

    def _update_pending_result(self):
        old = getattr(self, '_pending_result', None)
        if old is not None:
            self.canvas.remove_scene_item(old)
        self._pending_result = None
        if self.pending_spec and self.pending_spec.mode == 'anchored':
            w, h = self._cur_size()
            (x1, y1, x2, y2), _ = compute_crop(self.pending_spec, w, h, self.detections)
            it = QGraphicsRectItem(x1, y1, x2-x1, y2-y1)
            pen = QPen(QColor('#ffd000'), 2); pen.setStyle(Qt.DashLine)
            it.setPen(pen); it.setBrush(QBrush(QColor(0, 0, 0, 0))); it.setZValue(9)
            self.canvas.add_scene_item(it); self._pending_result = it

    def _clear_pending(self):
        if self.pending_item is not None:
            self.canvas.remove_scene_item(self.pending_item)
        pr = getattr(self, '_pending_result', None)
        if pr is not None:
            self.canvas.remove_scene_item(pr)
        self._pending_result = None
        self.pending_item = None; self.pending_spec = None
        self.define_editor.clear()

    def _redraw_pending(self):
        self._clear_pending()
        self._arm_next_draw()

    def _accept_pending(self):
        if self.pending_spec is None:
            return False
        self.specs.append(self.pending_spec)
        # drop the interactive item; accepted crops show as faint result outlines
        if self.pending_item is not None:
            self.canvas.remove_scene_item(self.pending_item)
        pr = getattr(self, '_pending_result', None)
        if pr is not None:
            self.canvas.remove_scene_item(pr)
        self._pending_result = None
        self.pending_item = None; self.pending_spec = None
        self.define_editor.clear()
        self._rebuild_strip()
        return True

    def _accept_next(self):
        if self._accept_pending():
            self._arm_next_draw()

    def _accept_finish(self):
        self._accept_pending()
        self._finish_define()

    def _finish_define(self):
        self._clear_pending()
        self.go_review()

    # accumulated preview --------------------------------------------------
    def _show_accepted_faint(self):
        w, h = self._cur_size()
        rects = []
        for s in self.specs:
            if s.side != self.current_side:
                continue
            (x1, y1, x2, y2), _ = compute_crop(s, w, h, self.detections)
            rects.append((x1, y1, x2, y2, s.name))
        self.canvas.show_faint(rects)

    def _rebuild_strip(self):
        while self.strip_layout.count() > 1:
            it = self.strip_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        w, h = self._cur_size()
        side_specs = [s for s in self.specs if s.side == self.current_side]
        for i, s in enumerate(side_specs, 1):
            (x1, y1, x2, y2), _ = compute_crop(s, w, h, self.detections)
            crop = self.img_bgr[y1:y2, x1:x2] if self.img_bgr is not None else None
            box = QVBoxLayout(); cell = QWidget(); cell.setLayout(box)
            thumb = QLabel(); thumb.setFixedSize(90, 70); thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet("border:1px solid #555")
            if crop is not None and crop.size:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ch, cw = rgb.shape[:2]
                qimg = QImage(rgb.data, cw, ch, 3*cw, QImage.Format_RGB888).copy()
                thumb.setPixmap(QPixmap.fromImage(qimg).scaled(90, 70, Qt.KeepAspectRatio,
                                                               Qt.SmoothTransformation))
            box.addWidget(thumb)
            lbl = QLabel(f"{i:02d} {s.name}"); lbl.setStyleSheet("color:#aaa; font-size:10px")
            box.addWidget(lbl)
            self.strip_layout.insertWidget(self.strip_layout.count()-1, cell)

    # =====================================================================
    # PAGE: REVIEW (edit all)
    # =====================================================================
    def _build_review(self):
        page = QWidget(); h = QHBoxLayout(page)
        self.review_canvas_holder = QVBoxLayout()
        h.addLayout(self.review_canvas_holder, 1)

        panel = QWidget(); panel.setFixedWidth(400); pv = QVBoxLayout(panel)
        top = QHBoxLayout()
        back = QPushButton("← Add more crops"); back.clicked.connect(lambda: self.go_define())
        top.addWidget(back)
        top.addWidget(QLabel("Side:"))
        self.review_side = QComboBox(); self.review_side.addItems(['front', 'back'])
        self.review_side.currentTextChanged.connect(self._on_review_side)
        top.addWidget(self.review_side)
        pv.addLayout(top)
        rc = QHBoxLayout()
        self.review_bill = QComboBox()
        for s in self.samples:
            self.review_bill.addItem(s)
        self.review_bill.currentTextChanged.connect(self._on_review_bill)
        rc.addWidget(QLabel("Preview bill:")); rc.addWidget(self.review_bill)
        self.review_overlay = QCheckBox("Elements"); self.review_overlay.setChecked(False)
        self.review_overlay.toggled.connect(lambda _: self._refresh_overlays())
        rc.addWidget(self.review_overlay)
        pv.addLayout(rc)

        self.review_list = QListWidget()
        self.review_list.currentRowChanged.connect(self._on_review_row)
        pv.addWidget(self.review_list)
        lrow = QHBoxLayout()
        addb = QPushButton("+ draw"); addb.clicked.connect(self._arm_review_add)
        delb = QPushButton("Delete"); delb.clicked.connect(self._delete_review)
        up = QPushButton("↑"); up.clicked.connect(lambda: self._move_review(-1))
        dn = QPushButton("↓"); dn.clicked.connect(lambda: self._move_review(1))
        for b in (addb, delb, up, dn):
            lrow.addWidget(b)
        pv.addLayout(lrow)

        self.review_editor = CropEditorWidget(self._cur_size, lambda: self.detections,
                                             self._on_review_changed)
        self.review_editor._img_fn = lambda: self.img_bgr
        pv.addWidget(self.review_editor)
        pv.addStretch()
        exp = QPushButton("Export collection (JSON)…"); exp.clicked.connect(self._export)
        pv.addWidget(exp)
        bt = QPushButton("Batch-crop a folder…"); bt.clicked.connect(self._open_batch)
        pv.addWidget(bt)
        h.addWidget(panel)
        return page

    def go_review(self):
        self.mode = 'edit'
        self.stack.setCurrentIndex(PAGE_REVIEW)
        self.review_canvas_holder.addWidget(self.canvas)
        self.canvas.clear_faint()
        self.review_side.blockSignals(True); self.review_side.setCurrentText(self.current_side)
        self.review_side.blockSignals(False)
        self._load_image()
        self._rebuild_review_items()

    def _rebuild_review_items(self):
        for it in list(self.crop_items.values()) + list(self.result_items.values()):
            self.canvas.remove_scene_item(it)
        self.crop_items.clear(); self.result_items.clear()
        self.review_list.clear()
        w, h = self._cur_size()
        for s in [s for s in self.specs if s.side == self.current_side]:
            self._add_review_item(s, w, h)
            self.review_list.addItem(f"{s.name}  [{s.mode}]")
        self._update_review_results()
        self.review_editor.clear()

    def _add_review_item(self, spec, w, h):
        b = spec.box
        item = CropRectItem(QRectF(b['x']*w, b['y']*h, b['w']*w, b['h']*h),
                            '#00d000' if spec.mode == 'anchored' else '#00b0ff')
        item._spec = spec
        item.on_geometry_change = self._on_review_geometry
        item.on_select = self._on_review_item_selected
        self.canvas.add_scene_item(item)
        self.crop_items[id(spec)] = item

    def _update_review_results(self):
        for s in [s for s in self.specs if s.side == self.current_side]:
            self._update_review_result(s)

    def _update_review_result(self, spec):
        old = self.result_items.pop(id(spec), None)
        if old is not None:
            self.canvas.remove_scene_item(old)
        if spec.mode != 'anchored':
            return
        w, h = self._cur_size()
        (x1, y1, x2, y2), _ = compute_crop(spec, w, h, self.detections)
        it = QGraphicsRectItem(x1, y1, x2-x1, y2-y1)
        pen = QPen(QColor('#ffd000'), 2); pen.setStyle(Qt.DashLine)
        it.setPen(pen); it.setBrush(QBrush(QColor(0, 0, 0, 0))); it.setZValue(9)
        self.canvas.add_scene_item(it); self.result_items[id(spec)] = it

    def _side_specs(self):
        return [s for s in self.specs if s.side == self.current_side]

    def _on_review_row(self, row):
        specs = self._side_specs()
        if row < 0 or row >= len(specs):
            self.selected_spec = None; self.review_editor.clear(); return
        self.selected_spec = specs[row]
        self.review_editor.load(self.selected_spec)
        for sid, it in self.crop_items.items():
            it.setSelected(sid == id(self.selected_spec))

    def _on_review_item_selected(self, item):
        spec = getattr(item, '_spec', None)
        if spec in self._side_specs():
            self.review_list.setCurrentRow(self._side_specs().index(spec))

    def _on_review_geometry(self, item):
        spec = item._spec; w, h = self._cur_size(); r = item.rect()
        spec.box = {'x': r.left()/w, 'y': r.top()/h, 'w': r.width()/w, 'h': r.height()/h}
        if spec.mode == 'anchored':
            d = derive_from_box((r.left(), r.top(), r.right(), r.bottom()), self.detections, w, h)
            if d:
                spec.anchor_classes = d['anchor_classes']
                spec.serial_pick = d['serial_pick']; spec.pad = d['pad']
        self._update_review_result(spec)
        if spec is self.selected_spec:
            self.review_editor.load(spec)

    def _on_review_changed(self, spec):
        it = self.crop_items.get(id(spec))
        if it is not None:
            it.set_color('#00d000' if spec.mode == 'anchored' else '#00b0ff')
        self._update_review_result(spec)
        specs = self._side_specs()
        if spec in specs:
            self.review_list.item(specs.index(spec)).setText(f"{spec.name}  [{spec.mode}]")

    def _arm_review_add(self):
        self.canvas.set_add_mode(True)

    def _add_review_crop(self, rect):
        w, h = self._cur_size()
        covers = (rect.width()*rect.height()) >= 0.85*(w*h)
        box_px = (rect.left(), rect.top(), rect.right(), rect.bottom())
        derived = None if covers else derive_from_box(box_px, self.detections, w, h)
        n = len(self._side_specs()) + 1
        spec = CropSpec(name=f"{self.current_side}_{n}", side=self.current_side,
                        box={'x': rect.left()/w, 'y': rect.top()/h,
                             'w': rect.width()/w, 'h': rect.height()/h})
        if derived:
            spec.mode = 'anchored'; spec.anchor_classes = derived['anchor_classes']
            spec.serial_pick = derived['serial_pick']; spec.pad = derived['pad']
        self.specs.append(spec)
        self._add_review_item(spec, w, h)
        self.review_list.addItem(f"{spec.name}  [{spec.mode}]")
        self.review_list.setCurrentRow(self.review_list.count()-1)
        self._update_review_result(spec)

    def _delete_review(self):
        if not self.selected_spec:
            return
        s = self.selected_spec
        for d in (self.crop_items, self.result_items):
            it = d.pop(id(s), None)
            if it is not None:
                self.canvas.remove_scene_item(it)
        self.specs.remove(s); self.selected_spec = None
        self._rebuild_review_items()

    def _move_review(self, delta):
        if not self.selected_spec:
            return
        specs = self._side_specs(); i = specs.index(self.selected_spec); j = i + delta
        if j < 0 or j >= len(specs):
            return
        a, b = specs[i], specs[j]
        ia, ib = self.specs.index(a), self.specs.index(b)
        self.specs[ia], self.specs[ib] = self.specs[ib], self.specs[ia]
        self._rebuild_review_items()
        self.review_list.setCurrentRow(j)

    def _on_review_side(self, side):
        self.current_side = side
        self.selected_spec = None
        self._load_image()
        self._rebuild_review_items()

    def _on_review_bill(self, serial):
        if serial in self.samples:
            self.current_sample = serial
            self._load_image()
            self._rebuild_review_items()

    # =====================================================================
    # shared: image load + overlays + batch + export
    # =====================================================================
    def _load_image(self):
        if not self.current_sample:
            return
        path = self.samples[self.current_sample][self.current_side]
        img = cv2.imread(path)
        if img is None:
            return
        self.img_bgr = img
        self.canvas.load_image(img)
        self._run_detection(path)
        self._refresh_overlays()

    def _run_detection(self, path):
        if not self.detector.available:
            self.detections = {}
            self._set_status("best.pt not found — fixed crops only"); return
        self._set_status("Detecting…"); QApplication.processEvents()
        try:
            self.detections = self.detector.detect(self.img_bgr, path)
            n = sum(len(v) for v in self.detections.values())
            self._set_status(f"{n} elements detected")
        except Exception as e:
            self.detections = {}; self._set_status(f"Detection error: {e}")

    def _set_status(self, msg):
        if hasattr(self, 'define_status') and self.stack.currentIndex() == PAGE_DEFINE:
            self.define_status.setText(msg)

    def _overlay_on(self):
        if self.stack.currentIndex() == PAGE_DEFINE:
            return self.define_overlay.isChecked()
        if self.stack.currentIndex() == PAGE_REVIEW:
            return self.review_overlay.isChecked()
        return False

    def _refresh_overlays(self):
        self.canvas.show_overlays(self.detections, self._overlay_on())

    def go_start(self):
        self.stack.setCurrentIndex(PAGE_START)

    def _open_batch(self):
        BatchDialog(self.specs, self.detector, self).exec()

    def _export(self):
        collection = {'name': 'Prototype Collection', 'active': True,
                      'crops': [self._export_spec(s) for s in self.specs]}
        path, _ = QFileDialog.getSaveFileName(
            self, "Export collection", str(Path.home() / "crop_collection.json"), "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(collection, indent=2))
        print(json.dumps(collection, indent=2))
        QMessageBox.information(self, "Exported", f"Wrote {len(self.specs)} crops to\n{path}")

    def _export_spec(self, spec):
        d = asdict(spec)
        if spec.mode != 'anchored':
            for k in ('anchor_classes', 'serial_pick', 'pad', 'min_width', 'min_height'):
                d.pop(k, None)
        return d


def main():
    app = QApplication(sys.argv)
    win = WizardController(Detector(DEFAULT_MODEL))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
