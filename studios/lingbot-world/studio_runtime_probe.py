#!/usr/bin/env python3
"""Probe a Studio runtime without reading environment-variable values or running inference."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def disk_fact(path: Path) -> dict[str, Any]:
    fact: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "writable_hint": os.access(path, os.W_OK) if path.exists() else False,
        "write_test": "not_requested",
    }
    if path.exists():
        usage = shutil.disk_usage(path)
        fact.update({
            "total_gib": round(usage.total / (1024 ** 3), 3),
            "used_gib": round(usage.used / (1024 ** 3), 3),
            "free_gib": round(usage.free / (1024 ** 3), 3),
        })
    return fact


def test_write(path: Path, fact: dict[str, Any]) -> None:
    if not path.is_dir():
        fact["write_test"] = "failed:not_a_directory"
        return
    try:
        handle = tempfile.NamedTemporaryFile(prefix="studio-probe-", dir=path, delete=False)
        probe_path = Path(handle.name)
        handle.write(b"probe")
        handle.close()
        probe_path.unlink()
        fact["write_test"] = "passed"
    except OSError as exc:
        fact["write_test"] = f"failed:{exc.__class__.__name__}"


def nvidia_smi() -> tuple[list[dict[str, Any]], str | None]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return [], "nvidia-smi not found"
    command = [binary, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    if result.returncode:
        return [], f"nvidia-smi failed with exit code {result.returncode}"
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        try:
            index: int | str = int(parts[0])
        except ValueError:
            index = parts[0]
        try:
            memory_mib: float | str = float(parts[2])
        except ValueError:
            memory_mib = parts[2]
        gpu: dict[str, Any] = {"index": index, "name": parts[1], "memory_total_mib": memory_mib, "driver_version": parts[3]}
        if isinstance(memory_mib, float):
            gpu["memory_total_gib"] = round(memory_mib / 1024, 3)
        gpus.append(gpu)
    return gpus, None


def torch_facts() -> tuple[dict[str, Any], str | None]:
    try:
        import torch  # type: ignore
    except (ModuleNotFoundError, ImportError) as exc:
        return {"installed": False}, f"PyTorch unavailable: {exc.__class__.__name__}"
    facts: dict[str, Any] = {
        "installed": True,
        "version": str(torch.__version__),
        "built_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            facts["cuda_devices"].append({
                "index": index,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "visible_memory_bytes": int(props.total_memory),
                "visible_memory_gib": round(props.total_memory / (1024 ** 3), 3),
                "allocated_gib": round(torch.cuda.memory_allocated(index) / (1024 ** 3), 4),
                "reserved_gib": round(torch.cuda.memory_reserved(index) / (1024 ** 3), 4),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated(index) / (1024 ** 3), 4),
            })
    return facts, None


def probe(workspace: Path, perform_write_test: bool) -> dict[str, Any]:
    workspace_fact = disk_fact(workspace)
    if perform_write_test:
        test_write(workspace, workspace_fact)
    root_fact = disk_fact(Path.cwd())
    gpus, smi_error = nvidia_smi()
    torch_info, torch_error = torch_facts()
    unknowns: list[dict[str, str]] = []
    if smi_error:
        unknowns.append({"id": "nvidia-smi-unavailable", "question": smi_error, "verification_path": "runtime.gpu_name"})
    if torch_error:
        unknowns.append({"id": "torch-unavailable", "question": torch_error, "verification_path": "runtime.torch_version"})
    elif not torch_info.get("cuda_available"):
        unknowns.append({"id": "torch-cuda-unavailable", "question": "PyTorch 未观察到可用 CUDA；当前是否为目标 xGPU 运行实例？", "verification_path": "runtime.visible_vram_gib"})
    if not workspace_fact["exists"]:
        unknowns.append({"id": "workspace-missing", "question": f"持久化目录 {workspace} 不存在", "verification_path": "runtime.persistent_workspace"})

    return {
        "schema_version": "1.0",
        "report_type": "runtime_probe",
        "generated_at": utc_now(),
        "tool": {"name": "runtime_probe.py", "version": VERSION},
        "target": {"hostname": socket.gethostname(), "working_directory": str(Path.cwd())},
        "facts": {
            "system": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or None,
            },
            "nvidia_smi_gpus": gpus,
            "torch": torch_info,
            "storage": {"working_directory": root_fact, "persistent_workspace": workspace_fact},
        },
        "findings": [],
        "unknowns": unknowns,
        "summary": {
            "nvidia_gpu_count": len(gpus),
            "torch_cuda_available": bool(torch_info.get("cuda_available")),
            "persistent_workspace_exists": bool(workspace_fact["exists"]),
            "write_test_performed": perform_write_test,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/mnt/workspace"), help="persistent workspace path")
    parser.add_argument("--test-write", action="store_true", help="create and immediately remove a temporary probe file")
    parser.add_argument("--output", type=Path, help="write JSON to this file instead of stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args()
    payload = json.dumps(probe(args.workspace, args.test_write), ensure_ascii=False, indent=None if args.compact else 2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
