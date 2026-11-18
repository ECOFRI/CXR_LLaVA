#!/usr/bin/env python
"""Colab-side EXP-02 C/D/F runner with hard gates and JSON logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def try_imports() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["torch", "transformers", "bitsandbytes", "accelerate", "tokenizers", "huggingface_hub", "PIL", "numpy", "pandas"]:
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            result[name] = f"import_error: {type(exc).__name__}: {exc}"
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        result["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as exc:
        result["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_command(command: list[str], cwd: str | None = None, timeout: int = 120) -> dict:
    started = utc_now()
    try:
        proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
        return {
            "command": command,
            "cwd": cwd,
            "start_utc": started,
            "end_utc": utc_now(),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-20000:],
        }
    except Exception as exc:
        return {
            "command": command,
            "cwd": cwd,
            "start_utc": started,
            "end_utc": utc_now(),
            "returncode": None,
            "exception": traceback.format_exc(),
        }


def probe_drive() -> dict:
    paths = [
        "/content/drive",
        "/content/drive/MyDrive",
        "/content/drive/MyDrive/ResearchLab/Notebook/datasets",
        "/content/drive/MyDrive/USTH_Master/ResearchLab/Notebook/datasets",
    ]
    result = {}
    for raw in paths:
        path = Path(raw)
        item = {"exists": path.exists(), "is_dir": path.is_dir()}
        if path.exists() and path.is_dir():
            try:
                item["children"] = sorted(p.name for p in list(path.iterdir())[:50])
            except Exception as exc:
                item["list_error"] = f"{type(exc).__name__}: {exc}"
        result[raw] = item
    return result


def resolve_dataset_root(exp_root: Path) -> dict:
    candidates = [
        Path("/content/drive/MyDrive/ResearchLab/Notebook/datasets"),
        Path("/content/drive/MyDrive/USTH_Master/ResearchLab/Notebook/datasets"),
    ]
    records = []
    selected = None
    for root in candidates:
        direct_png = root / "NLMCXR_png"
        direct_reports = root / "NLMCXR_reports"
        nested_png = root / "NLMCRX" / "NLMCXR_png"
        nested_reports = root / "NLMCRX" / "NLMCXR_reports"
        for png_dir, report_dir in [(direct_png, direct_reports), (nested_png, nested_reports)]:
            record = {
                "datasets_root": str(root),
                "png_dir": str(png_dir),
                "report_dir": str(report_dir),
                "png_exists": png_dir.exists(),
                "report_exists": report_dir.exists(),
            }
            if png_dir.exists():
                record["png_count"] = sum(1 for _ in png_dir.rglob("*.png"))
            if report_dir.exists():
                record["xml_count"] = sum(1 for _ in report_dir.rglob("*.xml"))
            records.append(record)
            if record["png_exists"] and record["report_exists"] and selected is None:
                selected = record
    result = {"candidates": records, "selected": selected}
    write_json(exp_root / "diagnostics" / "dataset_root_probe.json", result)
    if selected is None:
        raise RuntimeError("No Drive dataset root with NLMCXR_png and NLMCXR_reports was found.")
    return result


def phase_c_preflight(exp_root: Path) -> dict:
    exp_root.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "phase": "C",
        "start_utc": utc_now(),
        "environment": try_imports(),
        "drive_probe": probe_drive(),
    }
    write_json(exp_root / "logs" / "environment.json", diagnostics["environment"])
    write_json(exp_root / "diagnostics" / "drive_probe.json", diagnostics["drive_probe"])
    diagnostics["dataset_root"] = resolve_dataset_root(exp_root)
    diagnostics["end_utc"] = utc_now()
    diagnostics["status"] = "PASS_C1_DATASET_ROOT"
    write_json(exp_root / "logs" / "phase_c_preflight.json", diagnostics)
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    parser.add_argument("--exp-root", default="/content/exp02_cdf_probe")
    args = parser.parse_args()

    exp_root = Path(args.exp_root) / args.run_id
    status = {"run_id": args.run_id, "exp_root": str(exp_root), "start_utc": utc_now(), "status": "STARTED"}
    write_json(exp_root / "run_config.json", status)
    try:
        phase_c_preflight(exp_root)
        status["status"] = "STOP_AFTER_C1_BY_DESIGN"
        status["reason"] = "Phase C1 dataset root preflight completed. Model/template/retry/gate are intentionally separate gated steps."
    except Exception as exc:
        status["status"] = "BLOCKED"
        status["reason"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()
        print(status["traceback"], file=sys.stderr)
    finally:
        status["end_utc"] = utc_now()
        write_json(exp_root / "run_config.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
