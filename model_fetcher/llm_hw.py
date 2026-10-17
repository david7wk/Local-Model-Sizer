#!/usr/bin/env python3
"""
llm_hw.py
=========
Shared, dependency-free hardware detection for the local-LLM tool suite
(hf_model_sizer.py, llm_hardware_probe.py).

Python 3.8+ standard library only. Works on Windows 11, macOS
(Apple Silicon incl. M5, and Intel), and Linux.

Why this module exists
----------------------
The original scripts each carried their own copy of hardware detection, and
that copy had two blind spots that matter on modern laptops:

  1. On Windows, `nvidia-smi` is often NOT on PATH (Razer Blade 15, Lenovo
     Legion, etc.).  This module finds it anyway (PATH -> `where` ->
     %WINDIR%\\System32 -> Program Files\\NVIDIA Corporation\\NVSMI).
  2. On Windows laptops with hybrid graphics (Intel/AMD iGPU + NVIDIA dGPU),
     WMI knows about every display adapter even when the NVIDIA tool is
     missing.  Adapter VRAM is refined from the registry value
     HardwareInformation.qwMemorySize (WMI's AdapterRAM is capped at 4 GB).
  3. On Apple Silicon, GPU core count comes from `ioreg` (there is no
     reliable `sysctl hw.gpucores`), and chip-family memory bandwidth is
     estimated from a table of documented Apple values.

Public API
----------
  total_ram_bytes()            -> int | None
  avail_ram_bytes()            -> int | None
  nvidia_gpus()                -> list[dict]
  windows_display_gpus()       -> list[dict]
  apple_info()                 -> dict
  detect_budget(reserve, mac_reserve) -> (budget_gb, source_string)
  snapshot()                   -> dict (JSON-ready summary of everything)
"""

import ctypes
import os
import platform
import re
import shutil
import subprocess

GB = 1024 ** 3

# Rough compute-capability fallback for common NVIDIA GPUs when the driver
# does not report compute_cap (older drivers omit it from nvidia-smi).
CC_HINTS = [
    (r"GTX\s*16\d{2}", "7.5"),    # GTX 1650/1660 (Turing, no tensor cores)
    (r"GTX\s*10\d{2}", "6.1"),    # GTX 1050-1080 Ti (Pascal)
    (r"RTX\s*50\d{2}", "12.0"),   # RTX 50 series (Blackwell)
    (r"RTX\s*40\d{2}", "8.9"),    # RTX 40 series (Ada)
    (r"RTX\s*30\d{2}", "8.6"),    # RTX 30 series (Ampere)
    (r"RTX\s*20\d{2}", "7.5"),    # RTX 20 series (Turing)
    (r"TITAN\s*[RX]p?", "6.1"),
]

# Documented Apple Silicon memory bandwidth (GB/s). Chips not listed return
# None rather than a guess (e.g. M5 Pro/Max figures were not yet verified).
APPLE_BANDWIDTH_GBPS = {
    "M1": 68.25, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": 400, "M3 Ultra": 800,
    "M4": 120, "M4 Pro": 273, "M4 Max": 546,
    "M5": 153.6,
}


def run(cmd, timeout=20):
    """Run a command list and return stripped stdout, or None on any failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------- system RAM
def _windows_memory():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys, stat.ullAvailPhys


def total_ram_bytes():
    system = platform.system()
    try:
        if system == "Windows":
            return _windows_memory()[0]
        if system == "Darwin":
            out = run(["sysctl", "-n", "hw.memsize"])
            return int(out) if out and out.isdigit() else None
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def avail_ram_bytes():
    system = platform.system()
    try:
        if system == "Windows":
            return _windows_memory()[1]
        if system == "Darwin":
            out = run(["vm_stat"])
            if out:
                free = 0
                for line in out.splitlines():
                    m = re.match(r"^(Free pages|Pages free):\s+(\d+)", line)
                    if m:
                        free += int(m.group(2)) * 4096
                if free:
                    return free
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


# ------------------------------------------------------------------- NVIDIA
def find_nvidia_smi():
    """Locate nvidia-smi even when it is not on PATH (common on laptops)."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if platform.system() == "Windows":
        candidates = [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                         "System32", "nvidia-smi.exe"),
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        out = run(["where", "nvidia-smi"])
        if out:
            return out.splitlines()[0].strip()
    return None


def guess_cc(name):
    for pattern, cc in CC_HINTS:
        if re.search(pattern, name, re.I):
            return cc
    return "?"


def nvidia_gpus():
    """Query nvidia-smi for every NVIDIA GPU. Memory fields come back in MiB."""
    smi = find_nvidia_smi()
    if not smi:
        return []
    query = ("name,memory.total,memory.used,driver_version,compute_cap,"
             "utilization.gpu")
    out = run([smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    gpus = []
    for row in (out or "").splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 6:
            continue
        name, vram_mib, used_mib, driver, cc, util = parts[:6]
        try:
            vram_b = int(float(vram_mib) * 1024 ** 2)
            used_b = int(float(used_mib) * 1024 ** 2)
        except ValueError:
            continue
        if cc in ("", "[N/A]", "0", "0.0", "N/A"):
            cc = guess_cc(name)
        gpus.append({
            "name": name,
            "vram_bytes": vram_b,
            "used_bytes": used_b,
            "driver": driver,
            "compute_cap": cc,
            "util": util,
        })
    return gpus


# ---------------------------------------------------------------------- AMD
def amd_gpus():
    """AMD GPUs with VRAM where detectable.

    Linux: scan /sys/class/drm for PCI vendor 0x1002 and read the amdgpu
    sysfs counter mem_info_vram_total (dedicated VRAM; APUs report only their
    small carve-out). Names come from rocm-smi when installed.
    Windows: reuse the WMI/registry path, which reports real VRAM for
    discrete Radeon cards.
    """
    system = platform.system()
    gpus = []
    if system == "Linux":
        import glob
        rocm_name = None
        out = run(["rocm-smi", "--showproductname"], timeout=20)
        if out:
            m = re.search(r"Card series:\s*(.+)", out)
            if m:
                rocm_name = m.group(1).strip()
        for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
            try:
                with open(os.path.join(dev, "vendor")) as f:
                    if f.read().strip().lower() != "0x1002":
                        continue
            except OSError:
                continue
            vram = None
            try:
                with open(os.path.join(dev, "mem_info_vram_total")) as f:
                    vram = int(f.read().strip())
            except (OSError, ValueError):
                pass
            pci_id = ""
            try:
                with open(os.path.join(dev, "device")) as f:
                    pci_id = f.read().strip()
            except OSError:
                pass
            gpus.append({
                "name": rocm_name or f"AMD GPU (PCI {pci_id or '?'})",
                "vendor": "AMD",
                "vram_bytes": vram,
                "driver": "amdgpu",
                "status": "OK",
            })
    elif system == "Windows":
        gpus = [a for a in windows_display_gpus() if a.get("vendor") == "AMD"]
    return gpus


# ------------------------------------------------------------------ Windows
def windows_display_gpus():
    """All display adapters via WMI, VRAM refined from the registry.

    WMI's AdapterRAM field is a 32-bit value and tops out at ~4 GB, so for
    discrete cards we prefer the registry value HardwareInformation.qwMemorySize
    written by the display driver.
    """
    if platform.system() != "Windows":
        return []
    ps = ("Get-CimInstance Win32_VideoController | "
          "Select-Object Name,AdapterRAM,DriverVersion,Status | "
          "ConvertTo-Json -Compress")
    out = run(["powershell", "-NoProfile", "-Command", ps], timeout=30)
    if not out:
        return []
    import json
    try:
        data = json.loads(out)
    except ValueError:
        return []
    if isinstance(data, dict):          # ConvertTo-Json drops the array for 1 item
        data = [data]

    # Registry pass: DriverDesc -> real VRAM bytes (qwMemorySize / MemorySize)
    reg_vram = {}
    try:
        import winreg
        base = (r"SYSTEM\CurrentControlSet\Control\Class"
                r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
        i = 0
        while i < 64:
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE, f"{base}\\{i:04d}")
            except OSError:
                break
            with key:
                try:
                    desc, _ = winreg.QueryValueEx(key, "DriverDesc")
                except OSError:
                    i += 1
                    continue
                for val in ("HardwareInformation.qwMemorySize",
                            "HardwareInformation.MemorySize"):
                    try:
                        v, _ = winreg.QueryValueEx(key, val)
                        if isinstance(v, int) and v > 64 * 1024 ** 2:
                            reg_vram[desc.strip().lower()] = v
                            break
                    except OSError:
                        continue
            i += 1
    except Exception:
        pass

    gpus = []
    for d in data:
        name = str(d.get("Name", "")).strip()
        low = name.lower()
        if (not name or low == "microsoft basic display adapter"
                or "virtual" in low):
            continue
        vram = reg_vram.get(name.lower())
        if vram is None:
            try:
                vram = int(d.get("AdapterRAM") or 0) or None
            except (TypeError, ValueError):
                vram = None
        vendor = ("NVIDIA" if "nvidia" in name.lower()
                  else "AMD" if "amd" in name.lower() or "radeon" in name.lower()
                  else "Intel" if "intel" in name.lower()
                  else "Apple" if "apple" in name.lower() else "?")
        gpus.append({
            "name": name,
            "vendor": vendor,
            "vram_bytes": vram,           # None if undetectable
            "driver": str(d.get("DriverVersion") or ""),
            "status": str(d.get("Status") or ""),
        })
    return gpus


# -------------------------------------------------------------------- Apple
def apple_info():
    info = {}
    chip = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    info["chip"] = chip or "Apple Silicon (unknown)"
    info["model"] = run(["sysctl", "-n", "hw.model"]) or ""
    info["macos"] = run(["sw_vers", "-productVersion"]) or ""
    info["arch"] = platform.machine()
    info["apple_silicon"] = info["arch"] == "arm64"

    # GPU core count: no sysctl key exists; ioreg reports it for AGXAccelerator.
    out = run(["ioreg", "-rc", "AGXAccelerator"], timeout=30)
    counts = [int(x) for x in
              re.findall(r'"gpu-core-count" = (\d+)', out or "")]
    info["gpu_cores"] = max(counts) if counts else None

    # Performance-core count (hw.perflevel0 = performance cluster).
    pcores = run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
    info["perf_cores"] = int(pcores) if pcores and pcores.isdigit() else None

    # Chip family + documented memory bandwidth.
    m = re.search(r"Apple\s+(M\d+)\s*(Pro|Max|Ultra)?", info["chip"])
    info["chip_family"] = m.group(1) if m else None
    info["chip_variant"] = m.group(2) if m else None
    info["chip_name"] = (m.group(0) if m else None)
    info["mem_bandwidth_gbps"] = (
        APPLE_BANDWIDTH_GBPS.get(info["chip_name"])
        if info["chip_name"] else None)
    return info


# ------------------------------------------------------------- budget logic
def detect_budget(reserve=0.8, mac_reserve=6.0):
    """Return (usable_budget_gb, human-readable source).

    Order: NVIDIA VRAM minus reserve > Apple unified memory minus mac_reserve
    > 8 GB CPU-only fallback.
    """
    system = platform.system()

    gpus = nvidia_gpus()
    if gpus:
        best = max(gpus, key=lambda g: g["vram_bytes"])
        return (best["vram_bytes"] / GB - reserve,
                f"NVIDIA VRAM ({best['name']}) minus {reserve:g} GB reserve")

    # AMD discrete cards (>= 4 GB dedicated VRAM). APUs/iGPUs usually report
    # only a small carve-out here, so they are excluded from GPU budgets.
    amd = [a for a in amd_gpus() if (a.get("vram_bytes") or 0) >= 4 * GB]
    if amd:
        best = max(amd, key=lambda a: a["vram_bytes"])
        return (best["vram_bytes"] / GB - reserve,
                f"AMD VRAM ({best['name']}) minus {reserve:g} GB reserve "
                f"(Vulkan/ROCm engine)")

    if system == "Darwin":
        info = apple_info()
        ram = total_ram_bytes()
        if info["apple_silicon"] and ram:
            return (ram / GB - mac_reserve,
                    f"Apple unified memory minus {mac_reserve:g} GB for macOS")

    ram = total_ram_bytes()
    if ram and ram >= 32 * GB:
        return (16.0, "fallback: no accelerated GPU found; CPU-only estimate "
                      "(half of your 32+ GB RAM)")
    return (8.0, "fallback: no accelerated GPU found (CPU-only assumption)")


def snapshot():
    """Everything we can detect, as a JSON-ready dict."""
    snap = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_bytes": total_ram_bytes(),
        "avail_ram_bytes": avail_ram_bytes(),
    }
    nv = nvidia_gpus()
    if nv:
        snap["nvidia_gpus"] = nv
    amd = amd_gpus()
    if amd:
        snap["amd_gpus"] = amd
    if platform.system() == "Windows":
        snap["display_adapters"] = windows_display_gpus()
    if platform.system() == "Darwin":
        snap["apple"] = apple_info()
    budget, source = detect_budget()
    snap["budget_gb"] = round(budget, 2)
    snap["budget_source"] = source
    return snap



# ------------------------------------------------------- GGUF quant helpers
# Shared by hf_model_sizer.py and hf_recommend_and_fetch.py.

QUANT_RE = re.compile(
    r"(IQ[1-8]_[A-Z0-9]+|Q[2-8]_K_[A-Z]+|Q[2-8]_0|Q[2-8]_1|"
    r"Q\d+_\d+|BF16|FP16|F16|F32|MXFP4|4BIT|8BIT|6BIT|2BIT|3BIT)",
    re.I)

# Order matters: index is a quality rank (IQ1 first = lowest quality).
QUANT_RANK = ["IQ1", "Q2", "IQ2", "Q3", "IQ3", "Q4", "IQ4", "Q5", "IQ5",
              "Q6", "Q7", "Q8", "IQ8", "BF16", "FP16", "F16", "MXFP4"]


def quant_of(filename):
    m = QUANT_RE.search(filename)
    return m.group(1).upper() if m else "-"


def quant_rank(q):
    for i, prefix in enumerate(QUANT_RANK):
        if q.upper().startswith(prefix.upper()):
            return i
    return len(QUANT_RANK)


def verdict(file_gb, budget_gb):
    """GGUF size verdict vs budget: weights + ~15% runtime/KV allowance."""
    need = file_gb * 1.15
    if need <= budget_gb:
        return "FITS"
    if file_gb <= budget_gb:
        return "TIGHT (short context only)"
    if file_gb <= budget_gb * 1.9:
        return "STRETCH (partial CPU offload, 3-10x slower)"
    return "TOO BIG"


if __name__ == "__main__":
    import json
    print(json.dumps(snapshot(), indent=2))
