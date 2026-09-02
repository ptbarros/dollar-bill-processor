# Which version should I download?

The app ships in a few builds. They all do the exact same work and read the same
serials — they differ only in how they use your hardware to go faster. Pick the
one that matches your machine.

## Quick chooser

- **You have an NVIDIA graphics card** → **CUDA** build (`...-cuda-setup.exe`).
  Fastest by far, but a large (~1.9 GB) download because it bundles the NVIDIA
  runtime.
- **You have a normal Intel laptop/desktop** (most people) → **default** build
  (`...-setup.exe`, uses OpenVINO). Small download, and on Intel graphics from
  ~2020 on it's nearly as fast as CUDA.
- **You have an AMD or other non-Intel GPU and no NVIDIA card** → **DirectML**
  build (`...-directml-setup.exe`). Uses any DirectX 12 GPU.
- **Not sure?** Download the **default** build. It runs on everything and simply
  falls back to the CPU if it can't find a GPU it can use — it never fails to run.

You can install more than one build side by side; each has its own entry in the
Start menu and Add/Remove Programs.

## How fast is it? (measured on the same 100-bill batch)

| Your hardware | Build to download | ~100 bills | ≈ bills/min |
|---|---|---|---|
| NVIDIA RTX 5060 (or similar) | **CUDA** | ~34 s | ~180 |
| Intel Iris Xe — 11th-gen Core, e.g. i7-1185G7 | **Default (OpenVINO)** | ~65 s | ~90 |
| Intel UHD 630 — 9th-gen Core, e.g. i5-9500T | **Default (OpenVINO)** | ~3.5 min | ~27 |

These are whole-batch wall-clock times (including the one-time startup scan) on
real hardware. Your numbers will vary with CPU, scan resolution, and image count,
but the *relative* picture holds.

## What decides your speed

Two things run on every bill: **detection** (finding the serial) and **OCR**
(reading it). The build decides where detection runs:

- **NVIDIA GPU (CUDA):** both stages run on a powerful dedicated GPU — fastest.
- **Intel graphics (OpenVINO):** detection runs on the built-in Intel GPU, OCR on
  the CPU. How much this helps depends on how strong the Intel graphics are:
  - **Iris / Iris Xe (11th-gen Core and newer, ~2020+):** a big speedup — this is
    the sweet spot for the small default download.
  - **Older UHD graphics (roughly 6th–10th gen, ~2015–2019):** the built-in GPU is
    too weak to beat the CPU, so you get about CPU speed. It still works fine — just
    don't expect the Iris Xe numbers above.
  - **"F"-series Intel chips (e.g. i5-14400F) have no built-in graphics**, and AMD
    machines have no Intel GPU — these run on the CPU (via OpenVINO), still fine.
- **DirectML:** detection runs on any DirectX 12 GPU (AMD, older NVIDIA without a
  CUDA setup). OCR stays on the CPU.

**Bottom line:** the small default (OpenVINO) build is the right choice for almost
everyone. Only reach for the big CUDA build if you have an NVIDIA card and want the
absolute fastest processing.
