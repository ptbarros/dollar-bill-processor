# Explainer generators

Scripts that build the visual explainers in `docs/explainers/`. Run from the repo
root with the project venv active (needs `cv2`, the YOLO model `best.pt`, and
`ffmpeg` on PATH for the videos).

Paths are currently set for Paul's machine — the scan straps are read from
`~/Desktop/dollardetective/straps/*/results_*.csv` (edit the glob in
`render_seal_frames.py` if your scans live elsewhere).

## Seal Shift

```bash
# 1. Render the overlay frames (heavy — loads YOLO, runs across all straps).
#    Produces <out>/y/frame_###.png + manifest.json and <out>/x/... likewise.
python tools/explainers/render_seal_frames.py /tmp/seal_frames

# 2. Interactive page (self-contained; args: <frames> <out.html> [jpegQ] [maxWidth]).
python tools/explainers/build_html.py /tmp/seal_frames docs/explainers/seal-shift.html 70 620

# 3. Ping-pong MP4 + GIF for each sweep.
python tools/explainers/assemble_video.py /tmp/seal_frames docs/explainers/media
```

`render_seal_frames.py` picks a smooth sweep along one axis while holding the
other in a central band, registers each frame on the detected `ONE_hashed` box
(fixed position + scale via an affine warp), and draws the ONE box, the seal box,
the escaped (red) area, the shift vector, and the Seal&nbsp;X / Seal&nbsp;Y /
Containment readout. The metric geometry mirrors
`ProductionProcessor._calculate_seal_shift`.
