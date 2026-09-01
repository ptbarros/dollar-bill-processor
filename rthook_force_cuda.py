"""PyInstaller runtime hook for the CUDA edition.

Runs before the app imports anything. Forces the torch YOLO backend and the
EasyOCR backend (both GPU-capable via CUDA) instead of the default torch-free
ONNX + RapidOCR path. Uses setdefault so an explicit env override still wins.
"""
import os

os.environ.setdefault("DBP_FORCE_TORCH", "1")
os.environ.setdefault("DBP_OCR", "easy")
