"""Benchmark best.onnx across whatever ONNX Runtime providers are available.

Run this on the RTX Windows box to get the DirectML / CUDA GPU number we can't
measure on the CPU-only dev box.

Setup on the Windows machine (in the app's venv):
    pip install onnxruntime-directml        # GPU on any Windows GPU, no CUDA install
    # (or: pip install onnxruntime-gpu       # NVIDIA CUDA only)

Run:
    python tools/onnx_bench.py --onnx best.onnx --image path\\to\\front.jpg

It reports median inference latency for each available provider (CPU always,
plus DmlExecutionProvider and/or CUDAExecutionProvider if installed).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, imgsz=1792, stride=32):
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (imgsz - new_w) % stride, (imgsz - new_h) % stride
    dw, dh = dw / 2.0, dh / 2.0
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


def bench_provider(onnx_path, provider, blob, iters):
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    except Exception as e:
        return None, f"{provider}: unavailable ({type(e).__name__})"
    # confirm it actually bound to the requested provider
    active = sess.get_providers()[0]
    name = sess.get_inputs()[0].name
    # warmup
    for _ in range(3):
        sess.run(None, {name: blob})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: blob})
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2] * 1000
    return med, f"{provider} (active: {active})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="best.onnx")
    ap.add_argument("--image", required=True)
    ap.add_argument("--imgsz", type=int, default=1792)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    lb = letterbox(img, args.imgsz)
    blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
    print(f"Image {Path(args.image).name}  ->  net input {blob.shape}")
    print(f"Available providers: {ort.get_available_providers()}\n")

    # Try the GPU providers first, then CPU as the baseline.
    for provider in ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
        if provider not in ort.get_available_providers():
            continue
        med, label = bench_provider(args.onnx, provider, blob, args.iters)
        if med is None:
            print(f"  {label}")
        else:
            print(f"  {label:55s}  median {med:7.1f} ms")


if __name__ == "__main__":
    main()
