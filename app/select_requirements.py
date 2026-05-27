#!/usr/bin/env python3
"""Auto-detect GPU hardware and return the right requirements file.

Usage:
    python select_requirements.py        # prints filename (requirements.rocm.txt etc.)
    python select_requirements.py --install  # pip install -r <detected>
"""

import os
import sys
import subprocess
import argparse
import platform

HERE = os.path.dirname(os.path.abspath(__file__))


def detect():
    """Return ('rocm'|'nvidia'|'mps'|'cpu', reason_string)."""
    system = platform.system()

    # 1. Apple Silicon — check Metal Performance Shaders
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3
            )
            if "Apple" in out.stdout:
                return "mps", "Apple Silicon detected (MPS)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # system_profiler fallback
        try:
            out = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True, timeout=10
            )
            if "Apple" in out.stdout and ("M1" in out.stdout or "M2" in out.stdout or "M3" in out.stdout or "M4" in out.stdout):
                return "mps", "Apple Silicon detected via system_profiler"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # 2. Check for NVIDIA
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and "NVIDIA" in out.stdout:
            return "nvidia", "NVIDIA GPU detected via nvidia-smi"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Check for AMD ROCm
    rocm_paths = ["/opt/rocm/bin/rocm-smi", "/opt/rocm/bin/hipconfig"]
    for p in rocm_paths:
        if os.path.exists(p):
            return "rocm", f"AMD ROCm detected ({p} exists)"

    # 4. Fallback: check torch (if already installed)
    try:
        import torch
        if torch.cuda.is_available():
            if getattr(torch.version, "hip", None):
                return "rocm", "ROCm detected via torch (hip available)"
            return "nvidia", "CUDA detected via torch (nvidia-smi not in PATH)"
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "MPS detected via torch"
    except ImportError:
        pass

    # 5. CPU fallback
    return "cpu", "No GPU detected; falling back to CPU"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="Run pip install")
    args = parser.parse_args()

    kind, reason = detect()
    req_file = f"requirements.{kind}.txt"
    req_path = os.path.join(HERE, req_file)

    print(f"[detect] {reason}", file=sys.stderr)
    print(f"[detect] Selected: {req_file}", file=sys.stderr)

    if args.install:
        if not os.path.exists(req_path):
            print(f"[ERROR] {req_path} not found!", file=sys.stderr)
            sys.exit(1)
        cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
        print(f"[detect] Running: {' '.join(cmd)}", file=sys.stderr)
        subprocess.check_call(cmd)
    else:
        # Print just the filename for scripting
        print(req_file)


if __name__ == "__main__":
    main()