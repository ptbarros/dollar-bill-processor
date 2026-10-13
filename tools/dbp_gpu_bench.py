"""Self-contained GPU benchmark for best.onnx across ONNX Runtime providers.

Called by gpu_test.bat. Args:
    argv[1] = path to best.onnx
    argv[2] = optional path to a bill image (JPG). If missing/invalid, a
              synthetic bill-shaped tensor is used (latency is content-independent).

Writes a human-readable summary to stdout AND to gpu_test_result.txt next to the onnx.
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

IMGSZ = 1792
STRIDE = 32
ITERS = 25


def letterbox_blob(image_path):
    """Return a (1,3,H,W) float32 NCHW blob. Real image if given, else synthetic."""
    if image_path and Path(image_path).is_file():
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("cv2 could not read image")
            h, w = img.shape[:2]
            r = min(IMGSZ / h, IMGSZ / w)
            nw, nh = round(w * r), round(h * r)
            dw = (IMGSZ - nw) % STRIDE
            dh = (IMGSZ - nh) % STRIDE
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            top, bottom = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
            left, right = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
            blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            return np.ascontiguousarray(blob[None]), f"real image {Path(image_path).name} ({blob.shape[2]}x{blob.shape[1]})"
        except Exception as e:
            print(f"  (could not use image: {e}; falling back to synthetic)")
    # Synthetic bill-shaped tensor: 1792 wide, ~1$ bill aspect, stride-aligned.
    w = IMGSZ
    h = round((IMGSZ / 2.35) / STRIDE) * STRIDE  # ~768
    blob = np.random.rand(1, 3, h, w).astype(np.float32)
    return blob, f"synthetic tensor ({w}x{h})"


def bench(onnx_path, provider, blob):
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    except Exception as e:
        return None, None, f"unavailable ({type(e).__name__}: {e})"
    active = sess.get_providers()[0]
    if active != provider:
        # requested provider not really bound (e.g. DML not installed) -> skip
        return None, active, "not active (fell back to %s)" % active
    name = sess.get_inputs()[0].name
    for _ in range(3):
        sess.run(None, {name: blob})
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        sess.run(None, {name: blob})
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2] * 1000
    return med, active, "ok"


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "best.onnx"
    image = sys.argv[2] if len(sys.argv) > 2 else None
    if not Path(onnx_path).is_file():
        print(f"ERROR: {onnx_path} not found.")
        return 1

    blob, blob_desc = letterbox_blob(image)
    avail = ort.get_available_providers()

    lines = []
    lines.append("Dollar Bill Processor - GPU / provider benchmark")
    lines.append("=" * 52)
    lines.append(f"onnxruntime version : {ort.__version__}")
    lines.append(f"providers available : {', '.join(avail)}")
    lines.append(f"benchmark input     : {blob_desc}")
    lines.append(f"iterations          : {ITERS} (median reported)")
    lines.append("-" * 52)

    order = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    label = {"DmlExecutionProvider": "GPU (DirectML)",
             "CUDAExecutionProvider": "GPU (CUDA)",
             "CPUExecutionProvider": "CPU"}
    cpu_med = None
    gpu_med = None
    for prov in order:
        if prov not in avail:
            lines.append(f"{label[prov]:<18}: not available on this machine")
            continue
        med, active, status = bench(onnx_path, prov, blob)
        if med is None:
            lines.append(f"{label[prov]:<18}: skipped ({status})")
            continue
        lines.append(f"{label[prov]:<18}: {med:7.1f} ms / inference")
        if prov == "CPUExecutionProvider":
            cpu_med = med
        elif gpu_med is None:
            gpu_med = med

    if gpu_med and cpu_med:
        lines.append("-" * 52)
        lines.append(f"GPU speedup vs CPU  : {cpu_med / gpu_med:.1f}x faster")

    lines.append("=" * 52)
    report = "\n".join(lines)
    print(report)

    out = Path(onnx_path).resolve().parent / "gpu_test_result.txt"
    try:
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved a copy to: {out}")
        print("Send that file (gpu_test_result.txt) to Paul.")
    except Exception as e:
        print(f"(could not write result file: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
