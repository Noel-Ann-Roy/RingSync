"""
System information detection.

Pulls hardware/software context (CPU model, core counts, RAM, Python
and PyTorch versions, OS) so benchmark numbers can be interpreted
correctly -- a speedup curve means something different on a 4-core
laptop than on a 32-core workstation, and this makes that context
explicit rather than assumed.

Used by both the Streamlit System Information panel and the PDF report
export, so hardware context is identical in both places.
"""

import platform
import sys

import cpuinfo
import psutil
import torch


def _detect_windows_version(release: str) -> str:
   
    if platform.system() != "Windows":
        return release

    try:
        build = int(platform.version().split(".")[-1])
        if release == "10" and build >= 22000:
            return "11"
    except (ValueError, IndexError):
        pass
    return release


def get_system_info() -> dict:
   
    info = {}

    try:
        cpu_data = cpuinfo.get_cpu_info()
        info["cpu"] = cpu_data.get("brand_raw", "Unknown CPU")
    except Exception:
        info["cpu"] = platform.processor() or "Unknown CPU"

    try:
        info["logical_cores"] = str(psutil.cpu_count(logical=True))
    except Exception:
        info["logical_cores"] = str(__import__("os").cpu_count() or "Unknown")

    try:
        physical = psutil.cpu_count(logical=False)
        info["physical_cores"] = str(physical) if physical else "Unknown"
    except Exception:
        info["physical_cores"] = "Unknown"

    try:
        ram_bytes = psutil.virtual_memory().total
        info["ram_gb"] = f"{ram_bytes / (1024 ** 3):.0f} GB"
    except Exception:
        info["ram_gb"] = "Unknown"

    info["python_version"] = platform.python_version()

    try:
        info["pytorch_version"] = torch.__version__
    except Exception:
        info["pytorch_version"] = "Not installed"

    system = platform.system()
    if system == "Windows":
        release = _detect_windows_version(platform.release())
        info["os"] = f"Windows {release}"
    elif system == "Darwin":
        info["os"] = f"macOS {platform.mac_ver()[0]}"
    elif system == "Linux":
        info["os"] = f"Linux ({platform.release()})"
    else:
        info["os"] = system or "Unknown OS"

    return info


if __name__ == "__main__":
    info = get_system_info()
    for key, value in info.items():
        print(f"{key}: {value}")
