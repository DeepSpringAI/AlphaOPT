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


def _contains_experiment_error(value: Any) -> bool:
    if isinstance(value, str):
        return value == "experiment_error"
    if isinstance(value, list):
        return any(_contains_experiment_error(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_experiment_error(item) for item in value.values())
    return False


def is_repairable_evaluation_record(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return True
    result = record.get("result") or {}
    trace = record.get("trace") or {}
    status = result.get("status") or trace.get("first_status")
    attempts = trace.get("attempts")
    if status == "provider_policy_blocked":
        return False
    if status in {"experiment_error", "no_attempts"}:
        return True
    if attempts == [] and (trace.get("error") or status is None):
        return True
    return _contains_experiment_error(record.get("task_record"))


def repair_evaluation_state(state: dict[str, Any], *, enabled: bool = True) -> list[str]:
    """
    Drop corrupt/completed-error task records so resume re-runs them.

    The aggregate counters are rebuilt from the remaining completed records in-place.
    """
    if not enabled:
        return []
    completed = state.setdefault("completed_tasks", {})
    repaired = [key for key, record in list(completed.items()) if is_repairable_evaluation_record(record)]
    for key in repaired:
        completed.pop(key, None)

    aggregate = {
        "n_success": 0,
        "n_runnable": 0,
        "n_pass_at_k_success": 0,
        "n_pass_at_k_runnable": 0,
    }
    for record in completed.values():
        result = (record or {}).get("result") or {}
        aggregate["n_success"] += int(result.get("pass_at_1_success", 0) or 0)
        aggregate["n_runnable"] += int(result.get("pass_at_1_runnable", 0) or 0)
        aggregate["n_pass_at_k_success"] += int(result.get("pass_at_k_success", 0) or 0)
        aggregate["n_pass_at_k_runnable"] += int(result.get("pass_at_k_runnable", 0) or 0)
    state["aggregate"] = aggregate
    if repaired:
        state.setdefault("repair_history", []).append(
            {
                "repaired_at": now_timestamp(),
                "task_ids": repaired,
                "reason": "removed corrupt experiment-error records for resume",
            }
        )
        state["status"] = "in_progress"
        state["updated_at"] = now_timestamp()
    return repaired


def repair_training_state(
    state: dict[str, Any],
    tasks: Any | None = None,
    *,
    enabled: bool = True,
    batch_size: int | None = None,
) -> list[str]:
    """
    Remove completed training task records that were produced by infrastructure errors.

    For online learning, which resumes by batch boundary, this rewinds next_batch_start
    to the earliest corrupted task index when task records are available.
    """
    if not enabled:
        return []
    repaired: set[str] = set()

    for phase_name in ("diagnosis",):
        phase = state.get(phase_name) or {}
        for iter_state in phase.values():
            if not isinstance(iter_state, dict):
                continue
            task_results = iter_state.get("task_results") or {}
            bad_keys = [
                key
                for key, record in list(task_results.items())
                if _contains_experiment_error(record)
            ]
            for key in bad_keys:
                task_results.pop(key, None)
                repaired.add(key)
            if bad_keys:
                completed_ids = [task_key(task_id) for task_id in iter_state.get("completed_task_ids", [])]
                iter_state["completed_task_ids"] = [task_id for task_id in completed_ids if task_id not in set(bad_keys)]
                iter_state["status"] = "in_progress"

    online_state = state.get("online_learning") or {}
    if tasks is not None:
        task_list = list(tasks)
        corrupt_indices = [
            idx
            for idx, task in enumerate(task_list)
            if _contains_experiment_error(getattr(task, "output_status", []))
        ]
        if corrupt_indices:
            first_idx = min(corrupt_indices)
            effective_batch_size = int((batch_size or online_state.get("batch_size") or 1) or 1)
            rewind_to = (first_idx // effective_batch_size) * effective_batch_size
            online_state["next_batch_start"] = min(int(online_state.get("next_batch_start", 0) or 0), rewind_to)
            online_state["completed_task_ids"] = [task.id for task in task_list[:rewind_to]]
            online_state["status"] = "in_progress"
            repaired.update(task_key(task_list[idx].id) for idx in corrupt_indices)

    if repaired:
        state.setdefault("repair_history", []).append(
            {
                "repaired_at": now_timestamp(),
                "task_ids": sorted(repaired),
                "reason": "removed corrupt experiment-error training records for resume",
            }
        )
        state["status"] = "in_progress"
        state["updated_at"] = now_timestamp()
    return sorted(repaired)


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
