import os
import json
import time
import yaml
import statistics
from pathlib import Path
from typing import List, Tuple, Optional, Any

from tqdm.auto import tqdm

import concurrent.futures

from src import utils as alphaopt_utils
from src.utils import cal_time_cost, get_token_usage
from src.dataloader import DataLoader, Task          
from src.llm_programmer import ProgramGenerator
from src.experience_library import ExperienceLibrary
from src.llm_retriever import LibraryRetrieval
from src.train_eval_utils import check_optimality, self_debug 
from src.agent_tracing import agent_step, objective_attributes, record_event
from src.resume_state import (
    add_metric_totals,
    default_evaluation_state,
    load_json_state,
    now_timestamp,
    save_json_state,
    task_key,
)
from src.laminar_tracing import (
    add_span_tags,
    current_trace_id,
    flush_laminar,
    init_laminar_from_env,
    record_exception as laminar_record_exception,
    record_trace_index,
    set_span_attributes,
    set_span_output,
    set_trace_metadata,
    trace_span,
)

def evaluate(
    tasks: List["Task"],
    llm_opt: "ProgramGenerator",
    use_library: bool,
    library: Optional["ExperienceLibrary"],
    config: Any,
    *,
    resume_state_path: str,
    tasks_save_path: str,
    dataset: str,
    run_idx: int,
) -> Tuple[int, int, int, float, dict]:
    """
    Evaluate the task success rate of a learned experience library on a test dataset
    If use_library is False, the library is not used in the evaluation
    
    Returns:
        n_success: number of successful tasks (pass@1)
        n_runnable: number of runnable tasks
        n_total: total number of tasks
        pass_at_k_rate: pass@k success rate
        token_usage_delta: token usage delta for this evaluation
    """
    output_folder = config.output_folder
    pass_at_k = config.pass_at_k # Default to 1 if not specified

    # Track token usage before evaluation
    usage_before = get_token_usage()
    resume_enabled = bool(getattr(config, "resume", True))
    prior_state = (
        load_json_state(
            resume_state_path,
            default_evaluation_state(
                dataset=dataset,
                run_idx=run_idx,
                output_folder=output_folder,
                use_library=use_library,
            ),
        )
        if resume_enabled
        else default_evaluation_state(
            dataset=dataset,
            run_idx=run_idx,
            output_folder=output_folder,
            use_library=use_library,
        )
    )
    completed_tasks = prior_state.setdefault("completed_tasks", {})
    aggregate = prior_state.setdefault(
        "aggregate",
        {
            "n_success": 0,
            "n_runnable": 0,
            "n_pass_at_k_success": 0,
            "n_pass_at_k_runnable": 0,
        },
    )
    prior_token_usage_delta = prior_state.get("token_usage_delta_total", {})

    llm_retri = None

    if use_library:
        llm_retri = LibraryRetrieval(
            lib=library,
            model=llm_opt.model,
            service=config.service,
            temperature=llm_opt.temp,
        )

    def process_task(task, output_dirs):
        """
        Process a single task with multiple attempts for pass@k evaluation
        """
        task_trace_id = None
        output_path = output_dirs
        retrieved_ins_ids = []
        formulation_ins, program_ins = [], []
        with trace_span(
            "alphaopt.task",
            input={"task_id": task.id, "description": task.desc, "ground_truth": task.ground_truth},
            tags=[
                "alphaopt",
                "eval",
                str(dataset),
                str(llm_opt.model),
                "with-library" if use_library else "no-library",
            ],
            metadata={
                "mode": "evaluation",
                "phase": "evaluation",
                "dataset": dataset,
                "task_id": task.id,
                "run_idx": run_idx,
                "output_folder": output_folder,
                "model": llm_opt.model,
                "service": config.service,
                "library_path": getattr(config, "library_path", None),
                "taxonomy_path": getattr(config, "taxonomy_path", None),
                "resume_enabled": bool(getattr(config, "resume", True)),
            },
            attributes={
                "alphaopt.mode": "evaluation",
                "alphaopt.phase": "evaluation",
                "alphaopt.dataset": dataset,
                "alphaopt.task_id": str(task.id),
                "alphaopt.run_idx": int(run_idx),
                "alphaopt.use_library": bool(use_library),
                "alphaopt.ground_truth_objective": task.ground_truth,
            },
            session_id=f"evaluation:{output_folder}:run-{run_idx}",
        ) as root_span:
            try:
                task_trace_id = current_trace_id()
                set_trace_metadata(
                    {
                        "mode": "evaluation",
                        "phase": "evaluation",
                        "dataset": dataset,
                        "task_id": task.id,
                        "run_idx": run_idx,
                        "output_folder": output_folder,
                        "model": llm_opt.model,
                        "service": config.service,
                    }
                )
                if use_library:
                    # Retrieve relevant insights from an archived experience library
                    output_path = output_dirs
                    with agent_step(
                        "alphaopt.retrieval.formulation",
                        agent_name="LibraryRetrieval",
                        operation="retrieve_formulation_insights",
                        task=task,
                        dataset=dataset,
                        stage="Formulation",
                        input={"task_id": task.id, "stage": "Formulation"},
                        output_path=output_path,
                        attributes={"alphaopt.task_id": str(task.id), "alphaopt.retrieval.stage": "Formulation"},
                    ) as retr_span:
                        formulation_ins = llm_retri.retrieve_applicable_insights(
                            task=task,
                            stage="Formulation",
                            config=config,
                            verbose=False,
                            save_data=True,
                            output_path=output_path
                            )
                        retr_span.set_output({"retrieved_count": len(formulation_ins or [])})
                    retrieved_ins_ids = [ins["insight_id"] for ins in formulation_ins if 'insight_id' in ins]

                # Try multiple times for pass@k evaluation
                attempts_results = []
                attempt_records = []
                for attempt in range(pass_at_k):
                    attempt_output_path = f"{output_path}/attempt_{attempt + 1}" if pass_at_k > 1 else output_path
                    
                    with agent_step(
                        "alphaopt.formulation.generate",
                        agent_name="ProgramGenerator",
                        operation="generate_formulation",
                        task=task,
                        dataset=dataset,
                        stage="Formulation",
                        attempt=attempt + 1,
                        input={"task_id": task.id, "attempt": attempt + 1, "retrieved_count": len(formulation_ins or [])},
                        output_path=attempt_output_path,
                        attributes={"alphaopt.task_id": str(task.id), "alphaopt.attempt": attempt + 1},
                    ) as form_span:
                        candidate_model = llm_opt.generate_formulation(
                            task=task,
                            retrieved_insights=formulation_ins,
                            # rewrite=bool(config.ablation.rewrite),
                            abl_params=config.ablation,
                            verbose=False,
                            save_data=True,
                            output_path=attempt_output_path
                        )
                        form_span.set_output({"generated": bool(candidate_model)})
                    
                    if use_library and config.ablation.include_program_insight:
                        with agent_step(
                            "alphaopt.retrieval.program",
                            agent_name="LibraryRetrieval",
                            operation="retrieve_program_insights",
                            task=task,
                            dataset=dataset,
                            stage="Program",
                            attempt=attempt + 1,
                            input={"task_id": task.id, "attempt": attempt + 1, "stage": "Program"},
                            output_path=attempt_output_path,
                            attributes={"alphaopt.task_id": str(task.id), "alphaopt.attempt": attempt + 1, "alphaopt.retrieval.stage": "Program"},
                        ) as retr_span:
                            program_ins = llm_retri.retrieve_applicable_insights(
                                task=task,
                                stage="Program",
                                config=config,
                                formulation=candidate_model,
                                verbose=False,
                                save_data=True,
                                output_path=attempt_output_path
                            )
                            retr_span.set_output({"retrieved_count": len(program_ins or [])})
                        
                        retrieved_ins_ids.extend([ins["insight_id"] for ins in program_ins if "insight_id" in ins])

                    with agent_step(
                        "alphaopt.program.generate",
                        agent_name="ProgramGenerator",
                        operation="generate_program",
                        task=task,
                        dataset=dataset,
                        stage="Program",
                        attempt=attempt + 1,
                        input={"task_id": task.id, "attempt": attempt + 1, "retrieved_count": len(program_ins or [])},
                        output_path=attempt_output_path,
                        attributes={"alphaopt.task_id": str(task.id), "alphaopt.attempt": attempt + 1},
                    ) as prog_span:
                        candidate_program, output, runnable, is_time_out = llm_opt.generate_program(
                            task=task,
                            retrieved_insights=program_ins,
                            formulation=candidate_model,
                            abl_params=config.ablation,
                            verbose=False,
                            save_data=True,
                            output_path=attempt_output_path
                        )
                        prog_span.set_output({"generated": bool(candidate_program), "runnable": bool(runnable), "timeout": bool(is_time_out)})

                    # Check optimality
                    is_optimal, status, feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
                    
                    # Self-Debug
                    if config.ablation.max_debug_retry:
                        if status == "run_error":
                            is_optimal, runnable = self_debug(task, candidate_program, feedback, config, output_path=attempt_output_path)
                            record_event(
                                "self_debug_finished",
                                {
                                    "task_id": task.id,
                                    "attempt": attempt + 1,
                                    "is_optimal": bool(is_optimal),
                                    "runnable": bool(runnable),
                                },
                                output_path=attempt_output_path,
                            )

                    attempts_results.append((int(is_optimal), int(runnable), status))
                    attempt_records.append(
                        {
                            "attempt": attempt + 1,
                            "status": status,
                            "is_optimal": bool(is_optimal),
                            "runnable": bool(runnable),
                            "is_time_out": bool(is_time_out) if is_time_out is not None else None,
                            "output": str(output)[:2000],
                            "formulation_path": f"{attempt_output_path}/model_iter_None.txt",
                            "program_path": f"{attempt_output_path}/program_iter_None.py",
                            "output_path": f"{attempt_output_path}/output_iter_None.txt",
                        }
                    )
                    
                    # If we found a successful solution, we can stop early for pass@k
                    if is_optimal:
                        break

                if not attempts_results:
                    attempts_results.append((0, 0, "no_attempts"))
                    attempt_records.append({"attempt": 0, "status": "no_attempts", "is_optimal": False, "runnable": False})

                # Record task (use the first attempt's results for recording)
                task.retri_ins_lst.append(retrieved_ins_ids)
                task.output_status.append(attempts_results[0][2])  # Use first attempt's status

                # Calculate pass@k results
                pass_at_k_success = any(result[0] for result in attempts_results)  # Any attempt succeeded
                pass_at_k_runnable = any(result[1] for result in attempts_results)  # Any attempt was runnable
                final_status = attempts_results[0][2]
                best_objective = None
                for rec in attempt_records:
                    try:
                        best_objective = float(rec.get("output"))
                        break
                    except Exception:
                        continue
                final_obj_attrs = objective_attributes(
                    output=best_objective,
                    ground_truth=task.ground_truth,
                    matched=bool(pass_at_k_success),
                )
                set_span_attributes(root_span, {
                    "alphaopt.status": final_status,
                    "alphaopt.laminar_trace_id": task_trace_id,
                    "alphaopt.pass_at_k_success": bool(pass_at_k_success),
                    "alphaopt.pass_at_k_runnable": bool(pass_at_k_runnable),
                    **final_obj_attrs,
                })
                add_span_tags(root_span, [final_status, "optimal" if pass_at_k_success else "not-optimal"])
                trace_payload = {
                    "retrieved_insight_ids": retrieved_ins_ids,
                    "attempts": attempt_records,
                    "first_status": final_status,
                    "laminar_trace_id": task_trace_id,
                    **final_obj_attrs,
                }
                record_event(
                    "task_finished",
                    {
                        "task_id": task.id,
                        "status": final_status,
                        "pass_at_k_success": bool(pass_at_k_success),
                        "pass_at_k_runnable": bool(pass_at_k_runnable),
                        **final_obj_attrs,
                    },
                    output_path=output_path,
                )
                set_span_output(root_span, trace_payload)
                record_trace_index(
                    f"./testing/{output_folder}",
                    {
                        "mode": "evaluation",
                        "phase": "evaluation",
                        "dataset": dataset,
                        "task_id": task.id,
                        "run_idx": run_idx,
                        "trace_id": task_trace_id,
                        "status": final_status,
                        "artifact_path": output_dirs,
                    },
                )
                return (
                    int(attempts_results[0][0]),
                    int(attempts_results[0][1]),
                    int(pass_at_k_success),
                    int(pass_at_k_runnable),
                    trace_payload,
                )
            except Exception as exc:
                laminar_record_exception(root_span, exc)
                add_span_tags(root_span, ["experiment-error"])
                set_span_attributes(root_span, {"alphaopt.status": "experiment_error", "alphaopt.laminar_trace_id": task_trace_id})
                record_trace_index(
                    f"./testing/{output_folder}",
                    {
                        "mode": "evaluation",
                        "phase": "evaluation",
                        "dataset": dataset,
                        "task_id": task.id,
                        "run_idx": run_idx,
                        "trace_id": task_trace_id,
                        "status": "experiment_error",
                        "artifact_path": output_dirs,
                    },
                )
                raise

    def _restore_task_progress(task: "Task", record: dict[str, Any]) -> None:
        task.output_status = record.get("output_status", [])
        task.success_count = record.get("success_count", 0)
        task.confidence = record.get("success_confidence", record.get("confidence", 0))
        task.fail_to_execute = record.get("fail_to_execute", 0)
        task.fail_to_verify = record.get("fail_to_verify", 0)
        task.retri_ins_lst = record.get("retrieved_insights", [])

    for task in tasks:
        saved = completed_tasks.get(task_key(task.id))
        if saved and saved.get("task_record"):
            _restore_task_progress(task, saved["task_record"])

    output_dirs = [f"testing/{output_folder}/task_{task.id}" for task in tasks]
    pending = [
        (idx, task, output_dir)
        for idx, (task, output_dir) in enumerate(zip(tasks, output_dirs))
        if task_key(task.id) not in completed_tasks
    ]

    progress = tqdm(total=len(tasks), initial=len(completed_tasks), desc="Evaluating\n")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(process_task, task, output_dir): (idx, task)
                for idx, task, output_dir in pending
            }
            for future in concurrent.futures.as_completed(futures):
                idx, task = futures[future]
                try:
                    opt, run, pass_k, pass_k_run, trace_payload = future.result()
                except Exception as exc:
                    opt, run, pass_k, pass_k_run = 0, 0, 0, 0
                    trace_payload = {
                        "retrieved_insight_ids": [],
                        "attempts": [],
                        "first_status": "experiment_error",
                        "error": repr(exc),
                    }
                    task.output_status.append("experiment_error")
                key = task_key(task.id)
                completed_tasks[key] = {
                    "task_id": task.id,
                    "result": {
                        "pass_at_1_success": int(opt),
                        "pass_at_1_runnable": int(run),
                        "pass_at_k_success": int(pass_k),
                        "pass_at_k_runnable": int(pass_k_run),
                        "status": trace_payload.get("first_status"),
                    },
                    "trace": trace_payload,
                    "task_record": task.to_dict(mode="learn"),
                    "completed_at": now_timestamp(),
                }
                aggregate["n_success"] += int(opt)
                aggregate["n_runnable"] += int(run)
                aggregate["n_pass_at_k_success"] += int(pass_k)
                aggregate["n_pass_at_k_runnable"] += int(pass_k_run)
                current_delta = {}
                usage_after_partial = get_token_usage()
                for vendor, stats_after in usage_after_partial.items():
                    stats_before = usage_before.get(vendor, {})
                    current_delta[vendor] = {
                        k: float(stats_after.get(k, 0.0) - stats_before.get(k, 0.0))
                        for k in ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "cost")
                    }
                prior_state["token_usage_delta_total"] = add_metric_totals(prior_token_usage_delta, current_delta)
                prior_state["updated_at"] = now_timestamp()
                save_json_state(resume_state_path, prior_state)
                Path(tasks_save_path).parent.mkdir(parents=True, exist_ok=True)
                test_loader = DataLoader(task_list=list(tasks))
                test_loader.save_as_json(tasks_save_path)
                progress.update(1)
    finally:
        progress.close()

    n_success = int(aggregate.get("n_success", 0))
    n_runnable = int(aggregate.get("n_runnable", 0))
    n_pass_at_k_success = int(aggregate.get("n_pass_at_k_success", 0))
    n_pass_at_k_runnable = int(aggregate.get("n_pass_at_k_runnable", 0))
    
    pass_at_k_rate = n_pass_at_k_success / len(tasks) if len(tasks) > 0 else 0.0
    
    # Calculate token usage delta
    usage_after = get_token_usage()
    current_invocation_delta = {}
    for vendor, stats_after in usage_after.items():
        stats_before = usage_before.get(vendor, {})
        current_invocation_delta[vendor] = {
            k: float(stats_after.get(k, 0.0) - stats_before.get(k, 0.0))
            for k in ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "cost")
        }
    token_usage_delta = add_metric_totals(prior_token_usage_delta, current_invocation_delta)
    prior_state["token_usage_delta_total"] = token_usage_delta
    prior_state["status"] = "completed"
    prior_state["updated_at"] = now_timestamp()
    save_json_state(resume_state_path, prior_state)
    
    return n_success, n_runnable, len(tasks), pass_at_k_rate, token_usage_delta


def load_config(config_file: str) -> dict:
    """
    Load configuration from a YAML file
    """
    # with open(config_file, "r") as f:
    #     config = yaml.safe_load(f)  
    #* Configure
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_file)

    #* Generate a timestamp and append it to output_folder
    # ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # config.output_folder = f"{config.output_folder}_{ts}"
    # Re-resolve
    # OmegaConf.resolve(config)
    return config


def prepare_dataset_config(base_config: Any, dataset: str) -> Any:
    """
    Prepare configuration for a specific dataset by resolving template variables
    
    Args:
        base_config: Base configuration object
        dataset: Dataset name (must be a string)
        
    Returns:
        Dataset-specific configuration
    """
    from omegaconf import OmegaConf
    
    # Ensure dataset is a string
    dataset = str(dataset).strip()
    
    # Create a copy of the config (convert to dict first to avoid OmegaConf issues)
    config_dict = OmegaConf.to_container(base_config, resolve=False)
    
    # Remove 'datasets' field if it exists to avoid confusion
    if 'datasets' in config_dict:
        del config_dict['datasets']
    
    # Create new config from dict
    dataset_config = OmegaConf.create(config_dict)
    
    # Set dataset-specific values BEFORE resolving
    dataset_config.dataset = dataset
    
    # Resolve data_path and output_folder using the dataset name
    # Check for template fields first, then fall back to regular fields
    if 'data_path_template' in dataset_config:
        template = str(dataset_config.data_path_template)
        dataset_config.data_path = template.replace('${dataset}', dataset)
    elif 'data_path' in dataset_config:
        template = str(dataset_config.data_path)
        dataset_config.data_path = template.replace('${dataset}', dataset)
    else:
        dataset_config.data_path = f"./data/optimization_tasks/clean/{dataset}.json"
    
    if 'output_folder_template' in dataset_config:
        template = str(dataset_config.output_folder_template)
        dataset_config.output_folder = template.replace('${dataset}', dataset)
    elif 'output_folder' in dataset_config:
        template = str(dataset_config.output_folder)
        dataset_config.output_folder = template.replace('${dataset}', dataset)
    else:
        dataset_config.output_folder = f"{dataset}_new_flash"
    
    # Now resolve all variables (dataset is already set as a string)
    try:
        OmegaConf.resolve(dataset_config)
    except Exception as e:
        # If resolution fails, the manual replacements above should still work
        print(f"Warning: OmegaConf.resolve() failed: {e}, using manual replacements")
    
    return dataset_config


def evaluate_single_dataset(config: Any, dataset: str) -> dict:
    """
    Evaluate a single dataset
    
    Args:
        config: Base configuration
        dataset: Dataset name to evaluate
    """
    # Prepare dataset-specific configuration
    dataset_config = prepare_dataset_config(config, dataset)

    # Optional ablation tag (for output/log disambiguation)
    ablation_tag = str(getattr(dataset_config, "ablation_tag", "") or "").strip()
    if ablation_tag:
        # Make sure different ablation runs do not overwrite the same output folder
        dataset_config.output_folder = f"{dataset_config.output_folder}__{ablation_tag}"
    
    print(f"\n{'='*60}")
    print(f"Evaluating dataset: {dataset}")
    print(f"Data path: {dataset_config.data_path}")
    print(f"Output folder: {dataset_config.output_folder}")
    if ablation_tag:
        print(f"Ablation tag: {ablation_tag}")
    print(f"{'='*60}\n")

    # How many times to evaluate the same dataset (for mean/min/max)
    try:
        n_runs = int(getattr(dataset_config, "n_runs", 1) or 1)
    except Exception:
        n_runs = 1
    if n_runs < 1:
        n_runs = 1

    # Check if library_path is provided; if not, set use_library flag to False
    use_library = bool(dataset_config.library_path)

    if use_library:
        # Load trained experience library, optionally with the paired taxonomy file.
        library_path = str(dataset_config.library_path)
        taxonomy_path = str(getattr(dataset_config, "taxonomy_path", "") or "")

        if not os.path.isfile(library_path):
            raise FileNotFoundError(
                f"Configured library_path does not exist: {library_path}"
            )
        if taxonomy_path and not os.path.isfile(taxonomy_path):
            raise FileNotFoundError(
                f"Configured taxonomy_path does not exist: {taxonomy_path}"
            )

        print("Loading Library...")
        print(f"Library path: {library_path}")
        if taxonomy_path:
            print(f"Taxonomy path: {taxonomy_path}")

        library = ExperienceLibrary.from_json_file(
            library_path,
            taxonomy_path=taxonomy_path or None,
        )
        print(f"Loaded {len(library)} insights from the configured library.")
    else:
        print("Do task without Library...")
        library = None

    # Initialize ProgramGenerator
    reasoning_effort = getattr(dataset_config, "reasoning_effort", "medium")
    alphaopt_utils.config.reasoning_effort = reasoning_effort
    llm_opt = ProgramGenerator(
        model       = dataset_config.model,
        service     = dataset_config.service,
        temperature = dataset_config.temperature,
    )

    def _summarize(vals: List[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(statistics.mean(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    base_output_folder = str(dataset_config.output_folder)
    pass_at_k = dataset_config.pass_at_k

    per_run_results: List[dict] = []

    for run_idx in range(1, n_runs + 1):
        run_output_folder = f"{base_output_folder}/run_{run_idx}" if n_runs > 1 else base_output_folder
        dataset_config.output_folder = run_output_folder

        # Load test tasks fresh each run (avoid accumulating records in memory)
        test_tasks = DataLoader(dataset_config.data_path, mode="test")

        test_tasks_save_path = (
            f"./testing/{run_output_folder}/tasks_record_lib.json"
            if use_library
            else f"./testing/{run_output_folder}/tasks_record_nolib.json"
        )
        resume_state_path = f"./testing/{run_output_folder}/evaluation_resume_state.json"

        print(f"\n--- Run {run_idx}/{n_runs} ---")
        print(f"Output folder (run): {run_output_folder}")
        if bool(getattr(dataset_config, "resume", True)) and os.path.exists(resume_state_path):
            print(f"Resume state detected: {resume_state_path}")

        # Run evaluation
        start_time = time.time()
        n_success, n_runnable, n_total, pass_at_k_rate, token_usage_delta = evaluate(
            test_tasks,
            llm_opt,
            use_library,
            library,
            dataset_config,
            resume_state_path=resume_state_path,
            tasks_save_path=test_tasks_save_path,
            dataset=dataset,
            run_idx=run_idx,
        )
        success_rate = round(n_success / n_total, 3) if n_total else 0.0
        execution_rate = round(n_runnable / n_total, 3) if n_total else 0.0

        # Extract token cost (sum of all non-zero costs from all vendors)
        token_cost = sum(
            stats.get("cost", 0.0)
            for vendor, stats in token_usage_delta.items()
            if stats.get("cost", 0.0) != 0.0
        )

        # Extract token counts (sum across vendors)
        prompt_tokens = sum(float(stats.get("prompt_tokens", 0.0) or 0.0) for stats in token_usage_delta.values())
        completion_tokens = sum(float(stats.get("completion_tokens", 0.0) or 0.0) for stats in token_usage_delta.values())
        total_tokens = sum(float(stats.get("total_tokens", 0.0) or 0.0) for stats in token_usage_delta.values())

        # Count time cost (minutes as float)
        eval_duration = cal_time_cost(start_time, f"Evaluation for {dataset} (run {run_idx}/{n_runs})")

        print(
            f"\n================  EVALUATION RESULT ({dataset}) [Run {run_idx}/{n_runs}]  ================\n"
            f"Tasks evaluated : {n_total}\n"
            f"Pass@1 Success  : {n_success}\n"
            f"Pass@1 Rate     : {success_rate:.3%}\n"
            f"Pass@{pass_at_k} Rate     : {pass_at_k_rate:.3%}\n"
            f"Execution-rate  : {execution_rate:.3%}\n"
            f"Time cost (min) : {eval_duration}\n"
            f"Token cost      : ${token_cost:.6f}\n"
            f"====================================================\n"
        )

        # Save tasks with status record for this run
        test_tasks.save_as_json(test_tasks_save_path)

        # Store run metrics
        per_run_results.append(
            {
                "run_idx": run_idx,
                "output_folder": run_output_folder,
                "n_total": int(n_total),
                "n_success": int(n_success),
                "n_runnable": int(n_runnable),
                "pass_at_1_rate": float(success_rate),
                "pass_at_k_rate": float(pass_at_k_rate),
                "execution_rate": float(execution_rate),
                # Keep both keys for backward compatibility with older logs/scripts
                "duration_min": float(eval_duration),
                "duration": float(eval_duration),
                "token_cost": float(round(token_cost, 6)),
                "prompt_tokens": float(prompt_tokens),
                "completion_tokens": float(completion_tokens),
                "total_tokens": float(total_tokens),
                "token_usage_delta": token_usage_delta,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
            }
        )

    # Load existing logs if they exist, otherwise create a new list
    results_path = "./testing/all_test_results.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            if not isinstance(all_results, list):
                print(f"Warning: {results_path} is not a list JSON. Resetting it.")
                all_results = []
        except Exception as e:
            print(f"Warning: Failed to read {results_path}: {e}. Resetting it.")
            all_results = []
    else:
        all_results = []

    # Append per-run results
    for r in per_run_results:
        all_results.append(
            {
                "dataset": dataset_config.dataset,
                "data_path": dataset_config.data_path,
                "library_path": dataset_config.library_path if use_library else "None",
                "model": dataset_config.model,
                "service": dataset_config.service,
                "temperature": dataset_config.temperature,
                "reasoning_effort": reasoning_effort,
                "pass_at_k": pass_at_k,
                "n_runs": n_runs,
                "ablation_tag": ablation_tag or "base",
                "run_idx": r["run_idx"],
                "output_folder": r["output_folder"],
                "n_total": r["n_total"],
                "n_success": r["n_success"],
                "n_runnable": r["n_runnable"],
                "pass_at_1_rate": r["pass_at_1_rate"],
                "pass_at_k_rate": r["pass_at_k_rate"],
                "execution_rate": r["execution_rate"],
                "taxonomy": dataset_config.ablation.taxonomy,
                "rewrite": dataset_config.ablation.rewrite,
                "include_example": dataset_config.ablation.include_example,
                "include_program_insight": dataset_config.ablation.include_program_insight,
                "max_debug_retry": dataset_config.ablation.max_debug_retry,
                # Keep both keys for compatibility with existing `all_test_results.json`
                "duration_min": r["duration_min"],
                "duration": r.get("duration", r["duration_min"]),
                "token_cost": r["token_cost"],
                "prompt_tokens": r.get("prompt_tokens", 0.0),
                "completion_tokens": r.get("completion_tokens", 0.0),
                "total_tokens": r.get("total_tokens", 0.0),
                "timestamp": r["timestamp"],
            }
        )

    # Aggregate mean/min/max across runs
    agg = {
        "n_success": _summarize([float(r["n_success"]) for r in per_run_results]),
        "n_runnable": _summarize([float(r["n_runnable"]) for r in per_run_results]),
        "pass_at_1_rate": _summarize([float(r["pass_at_1_rate"]) for r in per_run_results]),
        "pass_at_k_rate": _summarize([float(r["pass_at_k_rate"]) for r in per_run_results]),
        "execution_rate": _summarize([float(r["execution_rate"]) for r in per_run_results]),
        "duration_min": _summarize([float(r["duration_min"]) for r in per_run_results]),
        "token_cost": _summarize([float(r["token_cost"]) for r in per_run_results]),
        "prompt_tokens": _summarize([float(r.get("prompt_tokens", 0.0)) for r in per_run_results]),
        "completion_tokens": _summarize([float(r.get("completion_tokens", 0.0)) for r in per_run_results]),
        "total_tokens": _summarize([float(r.get("total_tokens", 0.0)) for r in per_run_results]),
    }

    print(f"\n================  AGGREGATED RESULT ({dataset}) over {n_runs} run(s)  ================\n"
          f"Pass@1 Rate     : mean={agg['pass_at_1_rate']['mean']:.3%}, min={agg['pass_at_1_rate']['min']:.3%}, max={agg['pass_at_1_rate']['max']:.3%}\n"
          f"Pass@{pass_at_k} Rate     : mean={agg['pass_at_k_rate']['mean']:.3%}, min={agg['pass_at_k_rate']['min']:.3%}, max={agg['pass_at_k_rate']['max']:.3%}\n"
          f"Execution-rate  : mean={agg['execution_rate']['mean']:.3%}, min={agg['execution_rate']['min']:.3%}, max={agg['execution_rate']['max']:.3%}\n"
          f"Time cost (min) : mean={agg['duration_min']['mean']:.3f}, min={agg['duration_min']['min']:.3f}, max={agg['duration_min']['max']:.3f}\n"
          f"Token cost      : mean=${agg['token_cost']['mean']:.6f}, min=${agg['token_cost']['min']:.6f}, max=${agg['token_cost']['max']:.6f}\n"
          f"====================================================\n")

    # Append aggregate entry
    all_results.append(
        {
            "dataset": dataset_config.dataset,
            "data_path": dataset_config.data_path,
            "library_path": dataset_config.library_path if use_library else "None",
            "model": dataset_config.model,
            "service": dataset_config.service,
            "temperature": dataset_config.temperature,
            "reasoning_effort": reasoning_effort,
            "pass_at_k": pass_at_k,
            "n_runs": n_runs,
            "aggregate": True,
            "ablation_tag": ablation_tag or "base",
            "taxonomy": dataset_config.ablation.taxonomy,
            "rewrite": dataset_config.ablation.rewrite,
            "include_example": dataset_config.ablation.include_example,
            "include_program_insight": dataset_config.ablation.include_program_insight,
            "max_debug_retry": dataset_config.ablation.max_debug_retry,
            "summary": agg,
            "base_output_folder": base_output_folder,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())),
        }
    )

    # Save the updated log
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Restore output folder to base (avoid surprising callers)
    dataset_config.output_folder = base_output_folder

    return {
        "dataset": dataset_config.dataset,
        "data_path": dataset_config.data_path,
        "library_path": dataset_config.library_path if use_library else "None",
        "model": dataset_config.model,
        "service": dataset_config.service,
        "temperature": dataset_config.temperature,
        "reasoning_effort": reasoning_effort,
        "pass_at_k": pass_at_k,
        "n_runs": n_runs,
        "ablation_tag": ablation_tag or "base",
        "base_output_folder": base_output_folder,
        "summary": agg,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaOPT evaluation")
    parser.add_argument(
        "--config",
        default="./configs/eval/default.yaml",
        help="Path to evaluation config YAML file (default: ./configs/eval/default.yaml)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resume and ignore any existing evaluation checkpoint state.",
    )
    args = parser.parse_args()

    # Read the configuration file
    os.environ["ALPHAOPT_EVAL_CONFIG"] = args.config
    config = load_config(args.config)
    if args.no_resume:
        config.resume = False
    print(f"Evaluation config: {Path(args.config).resolve()}")
    init_laminar_from_env(mode="evaluation", config_path=args.config)

    # Get datasets list - support both single string and list
    # Check for 'datasets' first, then fall back to 'dataset' for backward compatibility
    if 'datasets' in config:
        datasets_raw = config.datasets
    elif 'dataset' in config:
        # Backward compatibility: if 'dataset' exists, convert to list
        datasets_raw = [config.dataset]
    else:
        raise ValueError("Configuration must contain either 'datasets' or 'dataset' field")
    
    # Convert to list and ensure all elements are strings
    # Handle OmegaConf ListConfig or regular list
    from omegaconf import ListConfig
    if isinstance(datasets_raw, (list, ListConfig)):
        datasets = [str(d) for d in datasets_raw]
    elif isinstance(datasets_raw, str):
        # If it's a string representation of a list, try to parse it
        if datasets_raw.startswith('[') and datasets_raw.endswith(']'):
            # This shouldn't happen with OmegaConf, but handle it just in case
            import ast
            try:
                datasets = [str(d) for d in ast.literal_eval(datasets_raw)]
            except:
                datasets = [datasets_raw]
        else:
            datasets = [datasets_raw]
    else:
        datasets = [str(datasets_raw)]

    print(f"\n{'='*60}")
    print(f"Starting batch evaluation for {len(datasets)} dataset(s)")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"{'='*60}\n")

    def _summarize(vals: List[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(statistics.mean(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    # Build ablation variants
    from omegaconf import OmegaConf
    from omegaconf import ListConfig
    base_cfg_container = OmegaConf.to_container(config, resolve=False)

    sweep_enabled = bool(getattr(config.ablation, "sweep_one_by_one", False))
    include_base = bool(getattr(config.ablation, "include_base", True))
    sweep_params = getattr(config.ablation, "sweep_params", None)
    if sweep_params is None:
        sweep_params = ["taxonomy", "include_example", "include_program_insight"]
    # OmegaConf uses ListConfig for YAML lists; treat it like a list here.
    sweep_params = [
        str(p)
        for p in (
            list(sweep_params)
            if isinstance(sweep_params, (list, tuple, ListConfig))
            else [sweep_params]
        )
    ]

    variants: List[tuple[str, dict]] = []
    if (not sweep_enabled) or include_base:
        variants.append(("base", {}))
    if sweep_enabled:
        for p in sweep_params:
            variants.append((f"{p}=false", {p: False}))

    # Run: variants × datasets
    all_variant_overalls: List[dict] = []

    def _append_to_results_log(entry: dict) -> None:
        """
        Best-effort append to ./testing/all_test_results.json
        """
        results_path = "./testing/all_test_results.json"
        try:
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = []
            else:
                all_results = []
        except Exception:
            all_results = []

        all_results.append(entry)
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write log to {results_path}: {e}")

    for v_i, (variant_name, overrides) in enumerate(variants, 1):
        print(f"\n{'#'*70}")
        print(f"Starting ablation variant [{v_i}/{len(variants)}]: {variant_name}")
        if overrides:
            print(f"Overrides: {overrides}")
        print(f"{'#'*70}\n")

        # Create variant config (deep-ish copy via container)
        v_cfg = OmegaConf.create(base_cfg_container)
        # Apply overrides
        for k, v in overrides.items():
            if hasattr(v_cfg.ablation, k):
                setattr(v_cfg.ablation, k, v)
            else:
                v_cfg.ablation[k] = v
        v_cfg.ablation_tag = variant_name

        all_dataset_aggs: List[dict] = []

        # Evaluate each dataset under this variant
        for i, dataset in enumerate(datasets, 1):
            dataset = str(dataset).strip()
            if dataset.startswith("[") and dataset.endswith("]"):
                print(f"\n⚠️  Warning: Dataset appears to be a list representation: {dataset}")
                print("   This suggests a configuration parsing issue. Skipping...")
                continue

            print(f"\n[{i}/{len(datasets)}] Processing dataset: {dataset} (variant: {variant_name})")
            try:
                dataset_agg = evaluate_single_dataset(v_cfg, dataset)
                if isinstance(dataset_agg, dict):
                    all_dataset_aggs.append(dataset_agg)
            except Exception as e:
                print(f"\n❌ Error evaluating dataset '{dataset}' (variant: {variant_name}): {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with next dataset...\n")
                continue

        print(f"\n{'='*60}")
        print(f"Variant completed: {variant_name} ({len(all_dataset_aggs)}/{len(datasets)} dataset(s) succeeded)")
        print(f"{'='*60}\n")

        # Per-variant overall summary across datasets (mean/min/max across dataset-level means)
        if all_dataset_aggs:
            def _get_mean_rate(key: str) -> List[float]:
                out: List[float] = []
                for d in all_dataset_aggs:
                    summary = d.get("summary") or {}
                    v = (summary.get(key) or {}).get("mean", None)
                    if v is not None:
                        out.append(float(v))
                return out

            overall = {
                "pass_at_1_rate": _summarize(_get_mean_rate("pass_at_1_rate")),
                "pass_at_k_rate": _summarize(_get_mean_rate("pass_at_k_rate")),
                "execution_rate": _summarize(_get_mean_rate("execution_rate")),
                "duration_min": _summarize(_get_mean_rate("duration_min")),
                "token_cost": _summarize(_get_mean_rate("token_cost")),
            }

            print(
                f"\n================  OVERALL AGGREGATED RESULT ({variant_name}) across {len(all_dataset_aggs)} dataset(s)  ================\n"
                f"Pass@1 Rate     : mean={overall['pass_at_1_rate']['mean']:.3%}, min={overall['pass_at_1_rate']['min']:.3%}, max={overall['pass_at_1_rate']['max']:.3%}\n"
                f"Pass@k Rate     : mean={overall['pass_at_k_rate']['mean']:.3%}, min={overall['pass_at_k_rate']['min']:.3%}, max={overall['pass_at_k_rate']['max']:.3%}\n"
                f"Execution-rate  : mean={overall['execution_rate']['mean']:.3%}, min={overall['execution_rate']['min']:.3%}, max={overall['execution_rate']['max']:.3%}\n"
                f"Time cost (min) : mean={overall['duration_min']['mean']:.3f}, min={overall['duration_min']['min']:.3f}, max={overall['duration_min']['max']:.3f}\n"
                f"Token cost      : mean=${overall['token_cost']['mean']:.6f}, min=${overall['token_cost']['min']:.6f}, max=${overall['token_cost']['max']:.6f}\n"
                f"====================================================================================\n"
            )

            all_variant_overalls.append(
                {
                    "variant": variant_name,
                    "n_datasets": len(all_dataset_aggs),
                    "datasets": [d.get("dataset") for d in all_dataset_aggs],
                    "summary": overall,
                }
            )

            _append_to_results_log(
                {
                    "aggregate_all_datasets": True,
                    "ablation_tag": variant_name,
                    "n_datasets": len(all_dataset_aggs),
                    "datasets": [d.get("dataset") for d in all_dataset_aggs],
                    "summary": overall,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())),
                }
            )

    # Final: multi-ablation summary across variants
    if all_variant_overalls:
        def _vmean(metric_key: str) -> List[float]:
            vals = []
            for v in all_variant_overalls:
                s = v.get("summary") or {}
                m = (s.get(metric_key) or {}).get("mean", None)
                if m is not None:
                    vals.append(float(m))
            return vals

        overall_over_variants = {
            "pass_at_1_rate": _summarize(_vmean("pass_at_1_rate")),
            "pass_at_k_rate": _summarize(_vmean("pass_at_k_rate")),
            "execution_rate": _summarize(_vmean("execution_rate")),
            "duration_min": _summarize(_vmean("duration_min")),
            "token_cost": _summarize(_vmean("token_cost")),
        }

        print(
            f"\n================  MULTI-ABLATION SUMMARY over {len(all_variant_overalls)} variant(s)  ================\n"
            f"Pass@1(mean over datasets) : mean={overall_over_variants['pass_at_1_rate']['mean']:.3%}, min={overall_over_variants['pass_at_1_rate']['min']:.3%}, max={overall_over_variants['pass_at_1_rate']['max']:.3%}\n"
            f"Pass@k(mean over datasets) : mean={overall_over_variants['pass_at_k_rate']['mean']:.3%}, min={overall_over_variants['pass_at_k_rate']['min']:.3%}, max={overall_over_variants['pass_at_k_rate']['max']:.3%}\n"
            f"Execution(mean over datasets): mean={overall_over_variants['execution_rate']['mean']:.3%}, min={overall_over_variants['execution_rate']['min']:.3%}, max={overall_over_variants['execution_rate']['max']:.3%}\n"
            f"Time(min, mean over datasets): mean={overall_over_variants['duration_min']['mean']:.3f}, min={overall_over_variants['duration_min']['min']:.3f}, max={overall_over_variants['duration_min']['max']:.3f}\n"
            f"Cost($, mean over datasets)  : mean=${overall_over_variants['token_cost']['mean']:.6f}, min=${overall_over_variants['token_cost']['min']:.6f}, max=${overall_over_variants['token_cost']['max']:.6f}\n"
            f"------------------------------------------------------------------------------------\n"
            f"{'variant':<26} {'P@1(mean)':>10} {'P@k(mean)':>10} {'Exec(mean)':>11} {'Time(min)':>10} {'Cost($)':>10}\n"
            f"------------------------------------------------------------------------------------"
        )
        for v in all_variant_overalls:
            s = v.get("summary") or {}
            p1 = (s.get("pass_at_1_rate") or {}).get("mean", 0.0)
            pk = (s.get("pass_at_k_rate") or {}).get("mean", 0.0)
            ex = (s.get("execution_rate") or {}).get("mean", 0.0)
            tm = (s.get("duration_min") or {}).get("mean", 0.0)
            co = (s.get("token_cost") or {}).get("mean", 0.0)
            print(f"{str(v.get('variant')):<26} {p1:>10.3%} {pk:>10.3%} {ex:>11.3%} {tm:>10.3f} {co:>10.6f}")
        print("====================================================================================\n")

        # Append multi-ablation summary to log (best-effort)
        _append_to_results_log(
            {
                "aggregate_all_variants": True,
                "n_variants": len(all_variant_overalls),
                "variants": [v.get("variant") for v in all_variant_overalls],
                "summary": overall_over_variants,
                "variant_overalls": all_variant_overalls,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())),
            }
        )
    flush_laminar()


if __name__ == "__main__":
    main()
