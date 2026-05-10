import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_json_state(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_state(path: str | Path, payload: Any) -> None:
    p = Path(path)
    ensure_parent_dir(p)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(p.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, p)


def task_key(task_id: Any) -> str:
    return str(task_id)


def add_metric_totals(base: dict[str, Any] | None, delta: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for source in (base or {}, delta or {}):
        for vendor, stats in source.items():
            vendor_totals = totals.setdefault(vendor, {})
            for key, value in (stats or {}).items():
                vendor_totals[key] = float(vendor_totals.get(key, 0.0) + float(value or 0.0))
    return totals


def default_evaluation_state(*, dataset: str, run_idx: int, output_folder: str, use_library: bool) -> dict[str, Any]:
    return {
        "mode": "evaluation",
        "dataset": str(dataset),
        "run_idx": int(run_idx),
        "output_folder": str(output_folder),
        "use_library": bool(use_library),
        "status": "in_progress",
        "completed_tasks": {},
        "aggregate": {
            "n_success": 0,
            "n_runnable": 0,
            "n_pass_at_k_success": 0,
            "n_pass_at_k_runnable": 0,
        },
        "token_usage_delta_total": {},
        "created_at": now_timestamp(),
        "updated_at": now_timestamp(),
    }


def default_training_state(*, config_path: str, library_subdir: str) -> dict[str, Any]:
    return {
        "mode": "training",
        "config_path": str(config_path),
        "library_subdir": str(library_subdir),
        "status": "in_progress",
        "current_phase": None,
        "current_iter": 0,
        "online_learning": {
            "next_batch_start": 0,
            "completed_task_ids": [],
            "status": "pending",
        },
        "diagnosis": {},
        "refinement": {},
        "created_at": now_timestamp(),
        "updated_at": now_timestamp(),
    }
