"""Phase 1 parity + speed harness: standalone ONNX Runtime vs. ultralytics/torch.

For each test image, run both backends at the confidence thresholds the app
actually uses (0.3 and 0.1) and compare detections per class:
  * same count of detections per class
  * matched boxes overlap with IoU >= --iou-match (default 0.98)
  * confidence differs by <= --conf-tol (default 0.02)

Also reports mean per-image latency for each backend (CPU).

Usage:
  venv/bin/python tools/onnx_parity.py --pt best.pt --onnx best.onnx --images <glob-or-dirs...>
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from yolo_backend import OnnxYoloDetector


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def torch_detect(model, img, conf):
    """Run ultralytics and normalize to (x1,y1,x2,y2,cls,conf) tuples."""
    res = model(img, verbose=False, conf=conf)
    out = []
    for r in res:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            out.append((x1, y1, x2, y2, int(box.cls[0]), float(box.conf[0])))
    return out


def compare(ref, test, names, iou_match, conf_tol):
    """Compare two detection lists. Returns (ok, list_of_problem_strings)."""
    problems = []
    by_cls_ref, by_cls_test = {}, {}
    for d in ref:
        by_cls_ref.setdefault(d[4], []).append(d)
    for d in test:
        by_cls_test.setdefault(d[4], []).append(d)

    for cls in set(by_cls_ref) | set(by_cls_test):
        R = by_cls_ref.get(cls, [])
        T = by_cls_test.get(cls, [])
        cname = names.get(cls, str(cls))
        if len(R) != len(T):
            problems.append(f"class {cname}: count {len(R)} (torch) vs {len(T)} (onnx)")
            continue
        # greedy match by IoU
        used = set()
        for rd in R:
            best_j, best_iou = -1, 0.0
            for j, td in enumerate(T):
                if j in used:
                    continue
                v = iou(rd[:4], td[:4])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j < 0 or best_iou < iou_match:
                problems.append(f"class {cname}: box IoU {best_iou:.4f} < {iou_match}")
                continue
            used.add(best_j)
            dconf = abs(rd[5] - T[best_j][5])
            if dconf > conf_tol:
                problems.append(f"class {cname}: conf diff {dconf:.4f} > {conf_tol}")
    return (len(problems) == 0), problems


def gather_images(patterns):
    files = []
    for p in patterns:
        pp = Path(p)
        if pp.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                files += [str(x) for x in pp.rglob(ext)]
        else:
            files += glob.glob(p)
    # de-dup, stable order
    seen, out = set(), []
    for f in sorted(files):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="best.pt")
    ap.add_argument("--onnx", default="best.onnx")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--confs", nargs="+", type=float, default=[0.3, 0.1])
    ap.add_argument("--iou-match", type=float, default=0.98)
    ap.add_argument("--conf-tol", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0, help="cap number of images")
    ap.add_argument("--imgsz", type=int, default=1792, help="inference long side")
    args = ap.parse_args()

    from ultralytics import YOLO
    tmodel = YOLO(args.pt)
    omodel = OnnxYoloDetector(args.onnx, imgsz=args.imgsz)

    images = gather_images(args.images)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print("No images found.")
        return
    print(f"Comparing {len(images)} images at confs {args.confs}\n")

    names = tmodel.names
    total, passed = 0, 0
    t_times, o_times = [], []

    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"  ! could not read {path}")
            continue
        for conf in args.confs:
            t0 = time.perf_counter()
            rt = torch_detect(tmodel, img, conf)
            t_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            ot = omodel.detect(img, conf=conf)
            o_times.append(time.perf_counter() - t0)

            ok, problems = compare(rt, ot, names, args.iou_match, args.conf_tol)
            total += 1
            passed += ok
            if not ok:
                print(f"FAIL  {Path(path).name}  conf={conf}")
                for pr in problems:
                    print(f"        - {pr}")

    print("\n" + "=" * 56)
    print(f"Parity: {passed}/{total} (image, conf) checks matched")
    # torch's first call includes warmup; report median to be fair
    def med(xs):
        return sorted(xs)[len(xs) // 2] * 1000 if xs else 0.0
    print(f"Median latency  torch/ultralytics: {med(t_times):6.1f} ms")
    print(f"Median latency  standalone onnx:    {med(o_times):6.1f} ms")
    print("=" * 56)


if __name__ == "__main__":
    main()
