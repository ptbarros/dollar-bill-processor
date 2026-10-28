"""Render scale/position-normalized zoom frames of the treasury seal vs the
ONE_hashed box (the "invisible box" the seal lives inside), for a seal-shift
explainer. Registered on the ONE box so the seal visibly drifts as the shift
sweeps. Produces a Y sweep and an X sweep.

Outputs: <out>/y/frame_###.png + manifest.json, <out>/x/... likewise
"""
import csv, glob, json, os, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path('/home/pbarros/projects/dollar-bill-processor')
sys.path.insert(0, str(ROOT))
from dollar_detective.process_production import ProductionProcessor, Config

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

# ---- gather bills ----
rows = []
for cf in glob.glob('/home/pbarros/Desktop/dollardetective/straps/*/results_*.csv'):
    strap = os.path.basename(os.path.dirname(cf))
    with open(cf) as f:
        for r in csv.DictReader(f):
            ff = r.get('front_file') or ''
            if not ff or not os.path.exists(ff):
                continue
            try:
                sx = float(r.get('seal_x') or 0); sy = float(r.get('seal_y') or 0)
                sc = float(r.get('seal_containment') or 100)
            except Exception:
                continue
            rows.append(dict(sx=sx, sy=sy, sc=sc, ff=ff, serial=r.get('serial', ''), strap=strap))
print(f"gathered {len(rows)} bills")


def select(axis):
    """Pick a smooth monotonic sweep along `axis` while holding the other axis
    in a central band, closest-to-target at each grid step, no repeats."""
    if axis == 'y':
        band = [r for r in rows if -5.5 <= r['sx'] <= -1.0]
        other_c = -3.2
        targets = [round(v * 0.25, 2) for v in range(int(-6 / 0.25), int(4 / 0.25) + 1)]
        key_t = lambda r, t: (abs(r['sy'] - t), abs(r['sx'] - other_c)); tol = 0.6
        near = lambda r, t: abs(r['sy'] - t) <= tol
    else:
        band = [r for r in rows if -2.0 <= r['sy'] <= 2.0]
        other_c = 0.0
        targets = [round(v * 0.3, 2) for v in range(int(-9 / 0.3), int(4 / 0.3) + 1)]
        key_t = lambda r, t: (abs(r['sx'] - t), abs(r['sy'] - other_c)); tol = 0.8
        near = lambda r, t: abs(r['sx'] - t) <= tol
    used, seq = set(), []
    for t in targets:
        cand = sorted((r for r in band if id(r) not in used), key=lambda r: key_t(r, t))
        if cand and near(cand[0], t):
            used.add(id(cand[0])); seq.append(cand[0])
    return seq


cfgp = ROOT / 'config.yaml'
P = ProductionProcessor(ROOT / 'best.pt', use_gpu=False, cfg=Config(cfgp if cfgp.exists() else None))

W, H = 720, 560
ONE_H = 250.0                 # ONE box height in output px (zoom level)
C_ONE = (90, 190, 90)         # BGR green -> ONE reference box
C_SEAL = (50, 110, 240)       # BGR orange -> treasury seal
C_VEC = (40, 180, 240)        # shift vector / amber


def tf(x, y, s, cx, cy):
    return (x - cx) * s + W / 2, (y - cy) * s + H / 2


def render(seq, sub):
    d = OUT / sub; d.mkdir(parents=True, exist_ok=True)
    man = []; fi = 0
    for r in seq:
        aligned, _ = P.align_for_preview(Path(r['ff']))
        if aligned is None:
            aligned = cv2.imread(r['ff'])
        if aligned is None:
            continue
        dets = P._detect_all_objects(aligned, conf=0.1)
        seals = [b for b in dets.get('seal_t', []) if b[4] >= 0.3]
        ones = [b for b in dets.get('ONE_hashed', []) if b[4] >= 0.3]
        if not seals or not ones:
            continue
        seal = max(seals, key=lambda b: b[4]); one = max(ones, key=lambda b: b[4])
        one_w, one_h = one[2] - one[0], one[3] - one[1]
        if one_w <= 0 or one_h <= 0:
            continue
        ocx, ocy = (one[0] + one[2]) / 2, (one[1] + one[3]) / 2
        scx, scy = (seal[0] + seal[2]) / 2, (seal[1] + seal[3]) / 2
        dx = (scx - ocx) / one_w * 100.0
        dy = -(scy - ocy) / one_h * 100.0
        ix1, iy1 = max(seal[0], one[0]), max(seal[1], one[1])
        ix2, iy2 = min(seal[2], one[2]), min(seal[3], one[3])
        sa = (seal[2] - seal[0]) * (seal[3] - seal[1])
        cont = (max(0, ix2 - ix1) * max(0, iy2 - iy1) / sa * 100.0) if sa else 0.0

        s = ONE_H / one_h
        M = np.array([[s, 0, W / 2 - s * ocx], [0, s, H / 2 - s * ocy]], dtype=np.float32)
        cv = cv2.warpAffine(aligned, M, (W, H), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(245, 245, 245))

        def C(x, y):
            return tuple(map(int, tf(x, y, s, ocx, ocy)))
        o1, o2 = C(one[0], one[1]), C(one[2], one[3])
        s1, s2 = C(seal[0], seal[1]), C(seal[2], seal[3])

        # (a) translucent green fill = the ONE "invisible box" the seal lives in
        ov = cv.copy(); cv2.rectangle(ov, o1, o2, C_ONE, -1)
        cv = cv2.addWeighted(ov, 0.12, cv, 0.88, 0)
        # (b) red tint on the part of the seal OUTSIDE the ONE box (-> containment)
        ov = cv.copy(); cv2.rectangle(ov, s1, s2, (40, 40, 235), -1)
        i1, i2 = C(ix1, iy1), C(ix2, iy2)
        if i2[0] > i1[0] and i2[1] > i1[1]:
            ov[i1[1]:i2[1], i1[0]:i2[0]] = cv[i1[1]:i2[1], i1[0]:i2[0]]
        cv = cv2.addWeighted(ov, 0.33, cv, 0.67, 0)
        # (c) outlines + centers + shift vector
        cv2.rectangle(cv, o1, o2, C_ONE, 2)
        cv2.rectangle(cv, s1, s2, C_SEAL, 2)
        oc, sc = C(ocx, ocy), C(scx, scy)
        cv2.drawMarker(cv, oc, C_ONE, cv2.MARKER_CROSS, 18, 2)
        cv2.drawMarker(cv, sc, C_SEAL, cv2.MARKER_CROSS, 18, 2)
        if abs(dx) + abs(dy) > 0.3:
            cv2.arrowedLine(cv, oc, sc, C_VEC, 2, tipLength=0.25)
        # labels
        cv2.putText(cv, "ONE box", (o1[0], max(18, o1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, C_ONE, 2, cv2.LINE_AA)
        cv2.putText(cv, "seal", (s2[0] + 6, s2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, C_SEAL, 2, cv2.LINE_AA)
        # top readout
        cv2.rectangle(cv, (0, 0), (W, 46), (28, 28, 28), -1)
        cv2.putText(cv, f"seal_y {dy:+.1f}%", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    C_SEAL if sub == 'y' else (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(cv, f"seal_x {dx:+.1f}%", (225, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    C_SEAL if sub == 'x' else (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(cv, f"containment {cont:.0f}%", (430, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (235, 235, 235) if cont >= 97 else (60, 120, 240), 2, cv2.LINE_AA)
        cv2.putText(cv, f"{r['serial']}  strap {r['strap']}", (16, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1, cv2.LINE_AA)

        fn = d / f"frame_{fi:03d}.png"
        cv2.imwrite(str(fn), cv)
        man.append(dict(i=fi, file=fn.name, seal_x=round(dx, 2), seal_y=round(dy, 2),
                        containment=round(cont, 1), serial=r['serial'], strap=r['strap']))
        fi += 1
        print(f"  [{sub}] frame {fi}: y={dy:+.2f} x={dx:+.2f} cont={cont:.0f} {r['serial']}")
    (d / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(f"  [{sub}] DONE {fi} frames")


for axis in ('y', 'x'):
    seq = select(axis)
    print(f"axis {axis}: selected {len(seq)} bills")
    render(seq, axis)
print("ALL DONE")
