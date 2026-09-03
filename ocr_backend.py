"""OCR backend abstraction: EasyOCR (default) or RapidOCR (torch-free).

Both backends expose the same call the pipeline already uses:

    reader.readtext(image, allowlist="ABC...", detail=1, whole_image=False)
        -> list of (bbox, text, conf)

EasyOCR is the default and unchanged. RapidOCR (onnxruntime, no torch) is opt-in
via DBP_OCR=rapid, and is the path toward dropping torch from the package.

Key design point: every call site except the last-resort full-image scan passes a
tight YOLO/contour crop of a single text region. RapidOCR's *text detector* tends
to over-segment such crops, so for crops we run it recognition-only (use_det=False)
— which reproduces EasyOCR's whole-line reads. The one full-image fallback passes
whole_image=True to enable RapidOCR's detector.

RapidOCR has no native allowlist, so it is emulated by filtering recognized text to
the allowed characters (uppercased). This is the main behavioral difference from
EasyOCR and the first thing to check if serial letters regress.
"""

import os


def _enable_openvino_for_rapidocr():
    """Prepend the OpenVINO EP to RapidOCR's session provider list (idempotent).

    RapidOCR only knows CPU/CUDA/DirectML, so we monkeypatch its OrtInferSession
    to add OpenVINOExecutionProvider when that wheel is present. No-op otherwise.
    """
    try:
        import onnxruntime as ort
        if "OpenVINOExecutionProvider" not in ort.get_available_providers():
            return None
        from rapidocr_onnxruntime.utils import infer_engine as ie
        cls = ie.OrtInferSession
        _dev = os.environ.get("DBP_OPENVINO_DEVICE_OCR", "CPU")
        if getattr(cls, "_dbp_openvino_patched", False):
            return f"OpenVINO:{_dev}"
        _orig = cls._get_ep_list
        # The small OCR models are FASTER on the CPU device than the iGPU (dispatch
        # overhead on tiny dynamic-shape inputs), so OCR defaults to CPU even when
        # YOLO uses the iGPU. DBP_OPENVINO_DEVICE_OCR overrides.
        def _get_ep_list(self):
            return [("OpenVINOExecutionProvider", {"device_type": _dev})] + list(_orig(self))

        cls._get_ep_list = _get_ep_list
        cls._dbp_openvino_patched = True
        return f"OpenVINO:{_dev}"
    except Exception:
        return None  # OCR must never fail to load over an optional speedup


def load_ocr_backend(use_gpu=False):
    """Return an OCR backend. RapidOCR (torch-free, faster, >= EasyOCR accuracy)
    is the default; EasyOCR is used only when DBP_OCR=easy, or as a fallback if
    RapidOCR fails to load."""
    if os.environ.get("DBP_OCR", "").lower() in ("easy", "easyocr"):
        return EasyOCRBackend(use_gpu=use_gpu)
    try:
        return RapidOCRBackend(use_gpu=use_gpu)
    except Exception as e:
        print(f"  RapidOCR unavailable ({type(e).__name__}: {e}); falling back to EasyOCR.")
        return EasyOCRBackend(use_gpu=use_gpu)


class EasyOCRBackend:
    """Thin wrapper over easyocr.Reader that ignores the whole_image hint."""

    name = "easyocr"

    def __init__(self, use_gpu=False):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    def readtext(self, image, allowlist="", detail=1, whole_image=False, detect=False, **kw):
        # whole_image/detect are RapidOCR routing hints; EasyOCR always detects.
        return self.reader.readtext(image, allowlist=allowlist, detail=detail, **kw)


class RapidOCRBackend:
    """RapidOCR (onnxruntime) presented with the EasyOCR readtext interface."""

    name = "rapidocr"

    def __init__(self, use_gpu=False):
        # When GPU acceleration is on and the OpenVINO wheel is installed, route
        # RapidOCR's ONNX sessions through OpenVINO (~2x faster on Intel CPUs).
        # RapidOCR has no native OpenVINO option, so we patch its EP list.
        self.device = _enable_openvino_for_rapidocr() if use_gpu else None
        self.device = self.device or "CPU"
        from rapidocr_onnxruntime import RapidOCR
        # RapidOCR bundles a lightweight (mobile) recognizer; DBP_RAPIDOCR_REC can
        # point at a heavier/English rec model (with DBP_RAPIDOCR_REC_KEYS for its
        # dict if it isn't the default Chinese one) to improve small-font digits.
        kwargs = {}
        rec = os.environ.get("DBP_RAPIDOCR_REC")
        if rec:
            kwargs["rec_model_path"] = rec
            keys = os.environ.get("DBP_RAPIDOCR_REC_KEYS")
            if keys:
                kwargs["rec_keys_path"] = keys
        self.engine = RapidOCR(**kwargs)

    @staticmethod
    def _filter(text, allowlist):
        text = text.upper()
        if allowlist:
            allowed = set(allowlist.upper())
            text = "".join(c for c in text if c in allowed)
        return text

    def readtext(self, image, allowlist="", detail=1, whole_image=False, detect=False, **kw):
        h, w = image.shape[:2]
        full_bbox = [[0, 0], [w, 0], [w, h], [0, h]]

        # Use RapidOCR's text detector for the full-image scan AND for multi-line /
        # multi-token regions (series year = "SERIES" + year, plates). Recognition-
        # only is reserved for genuinely single-line crops (the serial), where the
        # detector tends to over-segment.
        if whole_image or detect:
            # full detection + recognition
            res, _ = self.engine(image, use_det=True, use_cls=False, use_rec=True)
            rows = []
            for item in (res or []):
                bbox, text, conf = item[0], item[1], item[2]
                rows.append((bbox, text, float(conf)))
        else:
            # recognition-only: the caller already cropped one text region
            res, _ = self.engine(image, use_det=False, use_cls=False, use_rec=True)
            rows = [(full_bbox, t, float(c)) for (t, c) in (res or [])]

        out = []
        for bbox, text, conf in rows:
            ftext = self._filter(text, allowlist)
            if ftext:
                out.append((bbox, ftext, conf) if detail == 1 else ftext)
        return out
