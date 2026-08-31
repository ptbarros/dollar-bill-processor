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


def load_ocr_backend(use_gpu=False):
    """Return an OCR backend. RapidOCR when DBP_OCR=rapid and importable, else EasyOCR."""
    if os.environ.get("DBP_OCR", "").lower() in ("rapid", "rapidocr", "onnx"):
        try:
            return RapidOCRBackend(use_gpu=use_gpu)
        except Exception as e:
            print(f"  RapidOCR unavailable ({type(e).__name__}: {e}); using EasyOCR.")
    return EasyOCRBackend(use_gpu=use_gpu)


class EasyOCRBackend:
    """Thin wrapper over easyocr.Reader that ignores the whole_image hint."""

    name = "easyocr"

    def __init__(self, use_gpu=False):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    def readtext(self, image, allowlist="", detail=1, whole_image=False, **kw):
        return self.reader.readtext(image, allowlist=allowlist, detail=detail, **kw)


class RapidOCRBackend:
    """RapidOCR (onnxruntime) presented with the EasyOCR readtext interface."""

    name = "rapidocr"

    def __init__(self, use_gpu=False):
        from rapidocr_onnxruntime import RapidOCR
        # RapidOCR bundles its own ONNX models; CPU by default.
        self.engine = RapidOCR()

    @staticmethod
    def _filter(text, allowlist):
        text = text.upper()
        if allowlist:
            allowed = set(allowlist.upper())
            text = "".join(c for c in text if c in allowed)
        return text

    def readtext(self, image, allowlist="", detail=1, whole_image=False, **kw):
        h, w = image.shape[:2]
        full_bbox = [[0, 0], [w, 0], [w, h], [0, h]]

        if whole_image:
            # full detection + recognition (last-resort whole-image scan)
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
