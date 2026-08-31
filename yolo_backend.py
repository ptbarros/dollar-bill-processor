"""Inference backend for the bill-detection YOLO model.

Provides a standalone ONNX Runtime detector that reproduces ultralytics' YOLOv8
detections (no torch, no ultralytics at runtime) and is a drop-in replacement for
a loaded ``ultralytics.YOLO`` object: it is callable as
``detector(img, verbose=False, conf=X)`` and returns results whose ``.boxes``
iterate boxes exposing ``.cls[0]`` / ``.conf[0]`` / ``.xyxy[0]`` — the exact
interface every call site in process_production.py uses.

``load_detector()`` prefers the ONNX path when a sibling ``.onnx`` model and
onnxruntime are present, and transparently falls back to ultralytics/torch
otherwise (or when ``DBP_FORCE_TORCH=1``).

NOTE: dropping torch from the shipped package also requires replacing EasyOCR
(which imports torch); this backend removes torch only from YOLO inference.
"""

import ast
import os
from pathlib import Path

import cv2
import numpy as np

# Ultralytics defaults we must match for parity.
_MAX_WH = 7680      # class offset for class-aware (agnostic=False) NMS
_MAX_NMS = 30000    # candidate cap before NMS
_DEFAULT_IMGSZ = 1792


# --- ultralytics-compatible result shim (so call sites don't change) ---
class _Box:
    __slots__ = ("cls", "conf", "xyxy")

    def __init__(self, x1, y1, x2, y2, cls_id, conf):
        self.cls = (cls_id,)
        self.conf = (conf,)
        self.xyxy = ((x1, y1, x2, y2),)


class _Result:
    __slots__ = ("boxes",)

    def __init__(self, boxes):
        self.boxes = boxes


class OnnxYoloDetector:
    """Standalone onnxruntime YOLOv8 detector, callable like ultralytics YOLO."""

    def __init__(self, model_path, providers=None, imgsz=None):
        import onnxruntime as ort  # local import so the module loads without ORT

        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers or self._default_providers(ort),
        )
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

        meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.imgsz = imgsz or self._parse_imgsz(meta) or _DEFAULT_IMGSZ
        self.names = self._parse_names(meta)

    @staticmethod
    def _default_providers(ort):
        avail = ort.get_available_providers()
        order = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        return [p for p in order if p in avail] or ["CPUExecutionProvider"]

    @staticmethod
    def _parse_imgsz(meta):
        try:
            v = ast.literal_eval(meta.get("imgsz", ""))
            return int(v[0]) if isinstance(v, (list, tuple)) else int(v)
        except Exception:
            return None

    @staticmethod
    def _parse_names(meta):
        try:
            names = ast.literal_eval(meta.get("names", "{}"))
            return {int(k): v for k, v in names.items()}
        except Exception:
            return {}

    # -- ultralytics-equivalent rectangular letterbox (auto=True, scaleup, center) --
    def _letterbox(self, img, stride=32):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = round(w * r), round(h * r)
        dw = (self.imgsz - new_w) % stride
        dh = (self.imgsz - new_h) % stride
        dw /= 2.0
        dh /= 2.0
        if (w, h) != (new_w, new_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, dw, dh

    @staticmethod
    def _nms(boxes, scores, iou_thres):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_thres]
        return keep

    def detect(self, img_bgr, conf=0.25, iou=0.7, max_det=300):
        """Return a list of (x1, y1, x2, y2, cls_id, conf) in original pixels."""
        lb, r, dw, dh = self._letterbox(img_bgr)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        out = self.session.run(None, {self.input_name: blob})[0]  # [1, 4+nc, N]
        pred = out[0].T
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(1)
        confs = cls_scores.max(1)

        m = confs >= conf
        boxes_xywh, confs, cls_ids = boxes_xywh[m], confs[m], cls_ids[m]
        if boxes_xywh.shape[0] == 0:
            return []
        if confs.shape[0] > _MAX_NMS:
            top = confs.argsort()[::-1][:_MAX_NMS]
            boxes_xywh, confs, cls_ids = boxes_xywh[top], confs[top], cls_ids[top]

        xy, wh = boxes_xywh[:, :2], boxes_xywh[:, 2:4]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

        keep = self._nms(xyxy + cls_ids[:, None] * _MAX_WH, confs, iou)[:max_det]
        xyxy, confs, cls_ids = xyxy[keep], confs[keep], cls_ids[keep]

        xyxy[:, [0, 2]] -= dw
        xyxy[:, [1, 3]] -= dh
        xyxy /= r
        h0, w0 = img_bgr.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)

        return [
            (float(a), float(b), float(c), float(d), int(cid), float(cf))
            for (a, b, c, d), cid, cf in zip(xyxy, cls_ids, confs)
        ]

    # -- drop-in ultralytics call interface --
    def __call__(self, img_bgr, conf=0.25, verbose=False, **_ignored):
        dets = self.detect(img_bgr, conf=conf)
        return [_Result([_Box(*d) for d in dets])]


def load_detector(yolo_model_path, use_gpu=False):
    """Load the ONNX detector when possible, else fall back to ultralytics YOLO.

    Returns (detector, is_onnx). ONNX is chosen when a sibling ``.onnx`` model
    exists and onnxruntime imports, unless ``DBP_FORCE_TORCH=1`` is set.
    """
    pt_path = Path(yolo_model_path)
    onnx_path = pt_path.with_suffix(".onnx")
    force_torch = os.environ.get("DBP_FORCE_TORCH") == "1"

    if onnx_path.exists() and not force_torch:
        try:
            det = OnnxYoloDetector(onnx_path)
            return det, True
        except Exception as e:  # onnxruntime missing / bad model -> fall back
            print(f"  ONNX backend unavailable ({type(e).__name__}: {e}); using torch.")

    from ultralytics import YOLO
    return YOLO(str(pt_path)), False
