"""One-shot machine snapshot for the debug log.

Best-effort and dependency-light -- every probe is guarded and this module never
raises. It records the things that actually help when diagnosing a user's issue
or deciding whether a different Windows build (Standard / DirectML / NVIDIA)
would suit their machine better:

  * OS, Python, architecture
  * CPU model + core count, total RAM
  * GPU name(s)
  * the ONNX Runtime execution providers ACTUALLY available on that box
    (CUDA -> NVIDIA build viable, DirectML -> DirectML build, OpenVINO -> Standard)
  * the running edition and key package versions

Call ``log_system_info()`` once at startup; it writes a ``[SYSINFO]`` block to
debug_log.txt. Run it off the UI thread -- some probes shell out with a timeout.
"""

import os
import sys
import platform
import subprocess

# Hide the console window the subprocess probes would otherwise flash on Windows.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _run(cmd, timeout=5):
    """Run a command, return stdout or '' -- never raises, no visible window."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, creationflags=_NO_WINDOW)
        return out.stdout or ""
    except Exception:
        return ""


def _total_ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes

            class _MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MEM()
            m.dwLength = ctypes.sizeof(_MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1e9, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        return None


def _cpu_name():
    name = (platform.processor() or "").strip()
    try:
        if sys.platform == "win32" and (not name or "Family" in name):
            # wmic first (older Windows), PowerShell CIM as the fallback (11 24H2+).
            for line in _run(["wmic", "cpu", "get", "name"]).splitlines():
                line = line.strip()
                if line and line.lower() != "name":
                    return line
            ps = _run(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_Processor).Name"]).strip()
            if ps:
                return ps.splitlines()[0].strip()
        elif sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return name or "unknown"


def _gpu_names():
    names = []
    try:
        if sys.platform == "win32":
            for line in _run(["wmic", "path", "win32_VideoController", "get", "name"]).splitlines():
                line = line.strip()
                if line and line.lower() != "name":
                    names.append(line)
            if not names:
                ps = _run(["powershell", "-NoProfile", "-Command",
                           "(Get-CimInstance Win32_VideoController).Name"])
                names = [l.strip() for l in ps.splitlines() if l.strip()]
        elif sys.platform.startswith("linux"):
            for line in _run(["lspci"]).splitlines():
                if "VGA compatible controller" in line or "3D controller" in line:
                    names.append(line.split(":", 2)[-1].strip())
        elif sys.platform == "darwin":
            out = _run(["system_profiler", "SPDisplaysDataType"])
            for line in out.splitlines():
                if "Chipset Model:" in line:
                    names.append(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return names


def _onnx_providers():
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def _torch_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _edition():
    try:
        import updater
        return updater.detect_edition()
    except Exception:
        return "unknown"


def _pkg_version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def collect():
    """Return a dict of machine facts (best-effort; missing fields omitted)."""
    info = {"edition": _edition(),
            "os": platform.platform(),
            "arch": platform.machine(),
            "cpu": _cpu_name(),
            "cpu_cores": os.cpu_count(),
            "ram_gb": _total_ram_gb(),
            "gpu": "; ".join(_gpu_names()) or None,
            "torch_cuda_gpu": _torch_cuda(),
            "onnx_providers": ", ".join(_onnx_providers()) or None,
            "onnxruntime": _pkg_version("onnxruntime"),
            "torch": _pkg_version("torch"),
            "python": platform.python_version()}
    return {k: v for k, v in info.items() if v not in (None, "")}


def log_system_info():
    """Write a [SYSINFO] block to the debug log. Never raises; safe off-thread."""
    try:
        from debug_logger import dlog_raw
        info = collect()
        order = ("edition", "os", "arch", "cpu", "cpu_cores", "ram_gb", "gpu",
                 "torch_cuda_gpu", "onnx_providers", "onnxruntime", "torch", "python")
        lines = ["[SYSINFO] machine snapshot:"]
        for k in order:
            if k in info:
                lines.append(f"[SYSINFO]   {k}: {info[k]}")
        dlog_raw("\n".join(lines))
    except Exception:
        pass


if __name__ == "__main__":
    import json
    print(json.dumps(collect(), indent=2, default=str))
