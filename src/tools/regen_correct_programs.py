"""
Regenerate the `correct_program` field in the training JSON to use PuLP+CBC
instead of gurobipy.

Reuses ProgramGenerator.self_explore (src/llm_programmer.py): it iterates an LLM
up to 5 times, executes each candidate, and verifies the printed objective
against task.ground_truth. The retargeted PuLP prompts in src/prompts/prompts_opt.py
make the LLM emit PuLP code; the existing self_explore loop validates correctness.

Usage (preferred — see Makefile targets `regen-smoke` and `regen`):
    uv run python -m src.tools.regen_correct_programs \\
        --input  data/optimization_tasks/train/train_data_all_452.json \\
        --output data/optimization_tasks/train/train_data_all_452_pulp.json
"""

import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from pathlib import Path

# Allow `from src...` imports when invoked as a script (rather than `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omegaconf import OmegaConf

from src.dataloader import DataLoader
from src.llm_programmer import ProgramGenerator


SEED_FEEDBACK = (
    "The original program was written for Gurobi (`import gurobipy as gp`, "
    "`from gurobipy import GRB`). Translate it into a complete and runnable PuLP "
    "program that solves with the CBC backend, following the strict output format. "
    "Name the problem variable `model` and assign the result of `model.solve(...)` "
    "to `status`. Do NOT keep any gurobipy imports or GRB.* references."
)


def _regen_one(task, llm_opt, output_dir):
    """Run self_explore on a single task, returning (task_id, new_program_or_None)."""
    try:
        is_optimal, gold_program = llm_opt.self_explore(
            task=task,
            failed_program=task.correct_program or "",
            feedback=SEED_FEEDBACK,
            verbose=False,
            save_data=False,
            output_path=output_dir,
        )
        if is_optimal and gold_program:
            return task.id, gold_program, None
        return task.id, None, "self_explore did not converge to an optimal PuLP program"
    except Exception as exc:
        return task.id, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source training JSON containing Gurobi `correct_program` strings.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the regenerated JSON. Same shape as the input.",
    )
    parser.add_argument(
        "--config",
        default="train_config.yaml",
        help="Path to the YAML used to pick LLM model/service. Defaults to train_config.yaml.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel LLM calls. Lower this if you hit rate limits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of tasks to process (useful for smoke tests).",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)

    llm_opt = ProgramGenerator(
        model=config.advanced_model,
        service=config.advanced_service,
        temperature=0.7,
    )

    tasks = DataLoader(args.input, mode="learn", reset=False)
    if args.limit is not None:
        tasks = tasks.slice(0, args.limit)

    print(f"[regen] {len(tasks)} task(s) to process. Writing to {args.output}.")

    work_dir = Path(args.output).parent / "_regen_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_regen_one, task, llm_opt, str(work_dir)): task.id
            for task in tasks
        }
        for fut in concurrent.futures.as_completed(futures):
            task_id, program, err = fut.result()
            if program is not None:
                results[task_id] = program
                print(f"[regen] task {task_id}: ok")
            else:
                failures[task_id] = err
                print(f"[regen] task {task_id}: FAILED — {err}")

    # Write the regenerated dataset, preserving original ordering and fields.
    with open(args.input, "r", encoding="utf-8") as f:
        original = json.load(f)

    for item in original:
        tid = item.get("task_id")
        if tid in results:
            item["correct_program"] = results[tid]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(original, f, ensure_ascii=False, indent=2)

    print(
        f"[regen] done. {len(results)}/{len(tasks)} tasks regenerated. "
        f"{len(failures)} failures left unchanged in the output."
    )
    if failures:
        failure_log = Path(args.output).with_suffix(".failures.json")
        with open(failure_log, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"[regen] wrote failure details to {failure_log}.")


if __name__ == "__main__":
    main()
