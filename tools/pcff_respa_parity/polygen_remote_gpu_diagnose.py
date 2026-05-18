#!/usr/bin/env python3
"""Diagnose whether a remote worker can run the GROMACS GPU lane."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


COMMON_NVIDIA_SMI = [
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",
]


def run_cmd(cmd: list[str], timeout: int = 20) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "ok": proc.returncode == 0,
        }
    except Exception as exc:
        return {"cmd": cmd, "ok": False, "error": repr(exc)}


def first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def detect_nvidia_smi() -> str | None:
    path = shutil.which("nvidia-smi")
    if path:
        return path
    return first_existing(COMMON_NVIDIA_SMI)


def parse_cuda_release(text: str) -> tuple[int, int] | None:
    match = re.search(r"release\s+(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def cuda_release_at_least(release: tuple[int, int] | None, major: int, minor: int) -> bool:
    return release is not None and release >= (major, minor)


def cuda_candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value))
    roots.extend([Path("/usr/local/cuda"), Path("/usr")])
    return roots


def first_cuda_file(relative_paths: list[str]) -> str | None:
    for root in cuda_candidate_roots():
        for relative in relative_paths:
            candidate = root / relative
            if candidate.exists():
                return str(candidate)
    return None


def diagnose(repo: Path, gmx_gpu_binary: Path) -> dict[str, object]:
    nvidia_smi = detect_nvidia_smi()
    nvcc = shutil.which("nvcc")
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    devices = [path for path in ["/dev/nvidia0", "/dev/nvidiactl", "/dev/dxg"] if Path(path).exists()]

    nvidia_smi_result = run_cmd([nvidia_smi, "-L"]) if nvidia_smi else {"ok": False, "reason": "nvidia-smi not found"}
    nvcc_result = run_cmd([nvcc, "--version"]) if nvcc else {"ok": False, "reason": "nvcc not found"}
    gmx_result = run_cmd([str(gmx_gpu_binary), "--version"]) if gmx_gpu_binary.exists() else {
        "ok": False,
        "reason": f"missing {gmx_gpu_binary}",
    }
    disk_result = run_cmd(["df", "-h", str(repo)], timeout=10)
    mem_result = run_cmd(["free", "-h"], timeout=10)
    nproc_result = run_cmd(["nproc"], timeout=10)
    gmx_text = f"{gmx_result.get('stdout', '')}\n{gmx_result.get('stderr', '')}"
    gmx_reports_cuda = "CUDA" in gmx_text or "GPU support" in gmx_text
    nvcc_text = f"{nvcc_result.get('stdout', '')}\n{nvcc_result.get('stderr', '')}"
    nvcc_release = parse_cuda_release(nvcc_text)
    cufft_header = first_cuda_file(["include/cufft.h", "targets/x86_64-linux/include/cufft.h"])
    cufft_lib = first_cuda_file(
        [
            "lib/libcufft.so",
            "lib64/libcufft.so",
            "targets/x86_64-linux/lib/libcufft.so",
        ]
    )

    cuda_runtime_visible = bool(nvidia_smi_result.get("ok") or devices)
    cuda_build_inputs_available = bool(
        cmake
        and nvcc
        and cuda_release_at_least(nvcc_release, 12, 1)
        and cufft_header
        and cufft_lib
    )
    gmx_gpu_ready = bool(cuda_runtime_visible and gmx_result.get("ok") and gmx_reports_cuda)

    if gmx_gpu_ready:
        recommendation = "gmx_gpu lane can be smoke-tested."
    elif not cuda_runtime_visible:
        recommendation = "Expose an NVIDIA GPU to this worker first; WSL2 usually needs a Windows NVIDIA driver with WSL support and /dev/dxg or nvidia-smi visible inside WSL."
    elif not cuda_build_inputs_available:
        recommendation = "CUDA runtime is visible, but build inputs are incomplete; install/activate CUDA Toolkit >=12.1 with nvcc and cuFFT in the active conda env."
    elif not gmx_gpu_binary.exists():
        recommendation = "CUDA is visible, but GROMACS GPU binary is missing; run the remote setup notebook with RUN_BUILD_GMX_GPU=True."
    elif not gmx_reports_cuda:
        recommendation = "GROMACS binary exists but does not report CUDA/GPU support; rebuild with -DGMX_GPU=CUDA."
    else:
        recommendation = "GPU readiness is incomplete; inspect command outputs in this JSON."

    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.executable,
        "repo": str(repo),
        "gmx_gpu_binary": str(gmx_gpu_binary),
        "paths": {
            "nvidia_smi": nvidia_smi,
            "nvcc": nvcc,
            "cmake": cmake,
            "ninja": ninja,
            "devices": devices,
            "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            "CUDACXX": os.environ.get("CUDACXX", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "cufft_header": cufft_header,
            "cufft_lib": cufft_lib,
        },
        "versions": {
            "nvcc_release": list(nvcc_release) if nvcc_release else None,
        },
        "checks": {
            "cuda_runtime_visible": cuda_runtime_visible,
            "cuda_build_inputs_available": cuda_build_inputs_available,
            "gmx_gpu_binary_exists": gmx_gpu_binary.exists(),
            "gmx_reports_cuda": gmx_reports_cuda,
            "gmx_gpu_ready": gmx_gpu_ready,
        },
        "commands": {
            "nvidia_smi_L": nvidia_smi_result,
            "nvcc_version": nvcc_result,
            "gmx_gpu_version": gmx_result,
            "disk_free": disk_result,
            "memory": mem_result,
            "nproc": nproc_result,
        },
        "recommendation": recommendation,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("GROMACS_PCFF"))
    p.add_argument("--gmx-gpu-binary", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    gmx_gpu_binary = args.gmx_gpu_binary or (repo / "build_gateb_cuda/bin/gmx")
    result = diagnose(repo, gmx_gpu_binary.expanduser().resolve())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
        print(args.json_out)
    print(text)
    if args.strict and not result["checks"]["gmx_gpu_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
