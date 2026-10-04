"""
Draggable crop-box canvas for the Crop Manager.

Shows one sample-bill side and draws the selected crop region as a box you can
adjust directly on the bill, then emits the new rectangle in IMAGE-PIXEL coords so
the dialog can turn it into ``yolo_crops`` config via :mod:`crop_geometry`.

Two edit modes:
  * ``mode="box"`` (seals, serials, hand-placed fixed boxes): move the whole box
    and resize it from 8 corner/edge handles.
  * ``mode="edges"`` with ``edges`` (thirds): only the given vertical edges drag,
    for pulling the boundary between crops (overlap). The box itself doesn't move.

Handles are drawn at a constant on-screen size and have a generous grab margin (a
thin box outline is otherwise almost impossible to hit), and the cursor changes to
show what a drag will do.
"""

from __future__ import annotations
from typing import Optional, Tuple, Set

import numpy as np

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem,
)
from PySide6.QtGui import QPen, QColor, QBrush, QPixmap, QImage, QPainter, QPainterPath
from PySide6.QtCore import Qt, QRectF, Signal, QObject

Rect = Tuple[int, int, int, int]

_EDIT_COLOR = "#1e88e5"     # strong blue for the active box
_FAINT_COLOR = "#888888"    # muted grey for context outlines
_HANDLE_SCREEN = 11.0       # handle half-size, on-screen px (constant at any zoom)
_GRAB_SCREEN = 12.0         # extra grab margin, on-screen px

_CURSORS = {
    'tl': Qt.SizeFDiagCursor, 'br': Qt.SizeFDiagCursor,
    'tr': Qt.SizeBDiagCursor, 'bl': Qt.SizeBDiagCursor,
    'l': Qt.SizeHorCursor, 'r': Qt.SizeHorCursor,
    't': Qt.SizeVerCursor, 'b': Qt.SizeVerCursor,
    'move': Qt.OpenHandCursor,
}


class _CropRectItem(QGraphicsItem):
    """A crop rectangle the user can adjust. Geometry is in scene (image-pixel)
    coordinates. Calls ``on_live(rect)`` during a drag and ``on_change(rect)`` when
    it ends."""

    def __init__(self, rect: QRectF, color, on_change, on_live=None,
                 mode="box", edges: Optional[Set[str]] = None):
        super().__init__()
        self._rect = QRectF(rect).normalized()
        self._color = QColor(color)
        self._on_change = on_change
        self._on_live = on_live
        self._mode = mode                       # 'box' or 'edges'
        self._edges = set(edges or ())          # for 'edges' mode: which of l/r/t/b
        self._scale = 1.0                        # view px per scene unit
        self._drag = None
        self._start_scene = None
        self._start_rect = None
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    # --- metrics (kept constant on screen) -------------------------------
    def set_scale(self, scale: float):
        self.prepareGeometryChange()
        self._scale = max(1e-6, scale)
        self.update()

    @property
    def _hs(self):      # handle half-size in scene units
        return _HANDLE_SCREEN / self._scale

    @property
    def _mg(self):      # grab margin in scene units
        return _GRAB_SCREEN / self._scale

    # --- geometry --------------------------------------------------------
    def rect(self) -> QRectF:
        return QRectF(self._rect)

    def boundingRect(self) -> QRectF:
        m = self._hs + self._mg + 2
        return self._rect.adjusted(-m, -m, m, m)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        m = self._mg + self._hs
        if self._mode == "box":
            # whole (expanded) box is grabbable: interior moves, edges resize
            path.addRect(self._rect.adjusted(-m, -m, m, m))
        else:
            # only bands along the editable edges are grabbable
            r = self._rect
            for e in self._edges:
                if e == 'l':
                    path.addRect(QRectF(r.left() - m, r.top(), 2 * m, r.height()))
                elif e == 'r':
                    path.addRect(QRectF(r.right() - m, r.top(), 2 * m, r.height()))
                elif e == 't':
                    path.addRect(QRectF(r.left(), r.top() - m, r.width(), 2 * m))
                elif e == 'b':
                    path.addRect(QRectF(r.left(), r.bottom() - m, r.width(), 2 * m))
        return path

    def _handle_roles(self):
        return (('tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r')
                if self._mode == "box" else tuple(self._edges))

    def _corner_points(self):
        r = self._rect
        return {
            'tl': (r.left(), r.top()), 'tr': (r.right(), r.top()),
            'bl': (r.left(), r.bottom()), 'br': (r.right(), r.bottom()),
            't': (r.center().x(), r.top()), 'b': (r.center().x(), r.bottom()),
            'l': (r.left(), r.center().y()), 'r': (r.right(), r.center().y()),
        }

    def _role_at(self, pos):
        """Which handle/edge (or 'move') is under scene ``pos``, else None."""
        r = self._rect
        tol = self._hs + self._mg
        if self._mode == "box":
            pts = self._corner_points()
            # corners first, then edges (corners take priority)
            for role in ('tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r'):
                x, y = pts[role]
                if abs(pos.x() - x) <= tol and abs(pos.y() - y) <= tol:
                    # edge midpoints: require being near that edge line generally
                    if role in ('l', 'r') and not (r.top() - tol <= pos.y() <= r.bottom() + tol):
                        continue
                    if role in ('t', 'b') and not (r.left() - tol <= pos.x() <= r.right() + tol):
                        continue
                    return role
            if r.contains(pos):
                return 'move'
            return None
        # edges mode: proximity to an editable vertical/horizontal edge
        for e in self._edges:
            if e == 'l' and abs(pos.x() - r.left()) <= tol and r.top() <= pos.y() <= r.bottom():
                return 'l'
            if e == 'r' and abs(pos.x() - r.right()) <= tol and r.top() <= pos.y() <= r.bottom():
                return 'r'
            if e == 't' and abs(pos.y() - r.top()) <= tol and r.left() <= pos.x() <= r.right():
                return 't'
            if e == 'b' and abs(pos.y() - r.bottom()) <= tol and r.left() <= pos.x() <= r.right():
                return 'b'
        return None

    # --- painting --------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None):
        pen = QPen(self._color, 2)
        pen.setCosmetic(True)      # constant width regardless of zoom
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
        painter.drawRect(self._rect)

        hs = self._hs
        if self._mode == "box":
            painter.setBrush(QBrush(self._color))
            white = QPen(QColor('white'), 1); white.setCosmetic(True)
            painter.setPen(white)
            for x, y in self._corner_points().values():
                painter.drawRect(QRectF(x - hs, y - hs, 2 * hs, 2 * hs))
        else:
            # thick grips along the editable edges
            grip = QPen(self._color, 6); grip.setCosmetic(True)
            painter.setPen(grip)
            r = self._rect
            for e in self._edges:
                if e == 'l':
                    painter.drawLine(r.topLeft(), r.bottomLeft())
                elif e == 'r':
                    painter.drawLine(r.topRight(), r.bottomRight())
                elif e == 't':
                    painter.drawLine(r.topLeft(), r.topRight())
                elif e == 'b':
                    painter.drawLine(r.bottomLeft(), r.bottomRight())

    # --- interaction -----------------------------------------------------
    def hoverMoveEvent(self, ev):
        role = self._role_at(ev.scenePos())
        self.setCursor(_CURSORS.get(role, Qt.ArrowCursor))
        super().hoverMoveEvent(ev)

    def mousePressEvent(self, ev):
        self._drag = self._role_at(ev.scenePos())
        if self._drag is None:
            ev.ignore()
            return
        self._start_scene = ev.scenePos()
        self._start_rect = QRectF(self._rect)
        if self._drag == 'move':
            self.setCursor(Qt.ClosedHandCursor)
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        dx = ev.scenePos().x() - self._start_scene.x()
        dy = ev.scenePos().y() - self._start_scene.y()
        r = QRectF(self._start_rect)
        m = self._drag
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
        self.prepareGeometryChange()
        self._rect = r.normalized()
        self.update()
        if self._on_live:
            self._on_live(self.rect())
        ev.accept()

    def mouseReleaseEvent(self, ev):
        was = self._drag
        self._drag = None
        if was == 'move':
            self.setCursor(Qt.OpenHandCursor)
        if self._on_change:
            self._on_change(self.rect())
        ev.accept()


class CropCanvas(QGraphicsView):
    """Editable crop box over a sample bill.

    ``set_bill(img_bgr)`` then ``show_region(rect, ...)``. ``geometryChanged`` /
    ``geometryLive`` emit clamped image-pixel (x1,y1,x2,y2) tuples.
    """

    geometryChanged = Signal(tuple)
    geometryLive = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setMinimumHeight(200)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._edit_item: Optional[_CropRectItem] = None
        self._faint_items = []
        self._img_wh = (0, 0)

    # --- image -----------------------------------------------------------
    def set_bill(self, img_bgr: Optional[np.ndarray]):
        sc = self.scene()
        sc.clear()
        self._pixmap_item = None
        self._edit_item = None
        self._faint_items = []
        if img_bgr is None:
            self._img_wh = (0, 0)
            return
        import cv2
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._img_wh = (w, h)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self._pixmap_item.setZValue(0)
        sc.addItem(self._pixmap_item)
        sc.setSceneRect(0, 0, w, h)
        self._fit()

    def has_bill(self) -> bool:
        return self._pixmap_item is not None

    # --- regions ---------------------------------------------------------
    def clear_boxes(self):
        if self._edit_item is not None:
            self.scene().removeItem(self._edit_item)
            self._edit_item = None
        for it in self._faint_items:
            self.scene().removeItem(it)
        self._faint_items = []

    def show_region(self, rect: Optional[Rect], others=None, editable: bool = True,
                    mode: str = "box", edges=None):
        """Draw ``rect`` as the (optionally) editable box + ``others`` faint
        outlines. ``mode``/``edges`` pick box vs constrained-edge editing."""
        self.clear_boxes()
        from PySide6.QtWidgets import QGraphicsRectItem
        for orc in (others or []):
            if orc is None:
                continue
            it = QGraphicsRectItem(QRectF(orc[0], orc[1],
                                          orc[2] - orc[0], orc[3] - orc[1]))
            pen = QPen(QColor(_FAINT_COLOR), 2, Qt.DashLine); pen.setCosmetic(True)
            it.setPen(pen)
            it.setBrush(QBrush(QColor(0, 0, 0, 0)))
            it.setZValue(5)
            self.scene().addItem(it)
            self._faint_items.append(it)
        if rect is not None:
            qr = QRectF(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
            if editable:
                self._edit_item = _CropRectItem(
                    qr, _EDIT_COLOR, self._emit_final, self._emit_live,
                    mode=mode, edges=edges)
                self.scene().addItem(self._edit_item)
                self._sync_scale()
            else:
                it = QGraphicsRectItem(qr)
                pen = QPen(QColor(_EDIT_COLOR), 3); pen.setCosmetic(True)
                it.setPen(pen)
                it.setBrush(QBrush(QColor(0, 0, 0, 0)))
                it.setZValue(10)
                self.scene().addItem(it)
                self._faint_items.append(it)

    def _clamp(self, r: QRectF) -> Rect:
        w, h = self._img_wh
        x1 = max(0, min(int(round(r.left())), w))
        y1 = max(0, min(int(round(r.top())), h))
        x2 = max(0, min(int(round(r.right())), w))
        y2 = max(0, min(int(round(r.bottom())), h))
        return (x1, y1, x2, y2)

    def _emit_final(self, r: QRectF):
        self.geometryChanged.emit(self._clamp(r))

    def _emit_live(self, r: QRectF):
        self.geometryLive.emit(self._clamp(r))

    # --- fit -------------------------------------------------------------
    def _fit(self):
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._sync_scale()

    def _sync_scale(self):
        if self._edit_item is not None:
            self._edit_item.set_scale(abs(self.transform().m11()) or 1.0)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()
