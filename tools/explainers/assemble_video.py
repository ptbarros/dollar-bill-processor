"""Assemble ping-pong MP4 + GIF for each sweep from the rendered PNG frames."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

FRAMES = Path(sys.argv[1])          # .../seal_frames
OUTDIR = Path(sys.argv[2])          # where mp4/gif land
OUTDIR.mkdir(parents=True, exist_ok=True)
FPS = 9

NAMES = {'y': 'seal-shift-vertical', 'x': 'seal-shift-horizontal'}

for axis, stem in NAMES.items():
    d = FRAMES / axis
    man = json.loads((d / 'manifest.json').read_text())
    files = [d / m['file'] for m in man]
    # ping-pong: forward then back (drop shared endpoints)
    order = files + files[-2:0:-1]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, f in enumerate(order):
            shutil.copy(f, tmp / f"{i:04d}.png")
        seq = str(tmp / "%04d.png")
        mp4 = OUTDIR / f"{stem}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", seq,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
            "-movflags", "+faststart", str(mp4)], check=True)
        # gif via palette (downscaled for size)
        pal = tmp / "pal.png"
        vf = "fps=8,scale=440:-2:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", seq,
                        "-vf", vf + ",palettegen=stats_mode=diff", str(pal)], check=True)
        gif = OUTDIR / f"{stem}.gif"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", seq, "-i", str(pal),
                        "-lavfi", vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        str(gif)], check=True)
        print(f"{stem}: {len(order)} frames  mp4 {mp4.stat().st_size/1e6:.2f}MB  "
              f"gif {gif.stat().st_size/1e6:.2f}MB")
print("DONE")
