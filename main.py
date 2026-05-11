import os
import time
import json
import argparse
from pathlib import Path
from src.config import (
    get_train_config_path,
    prepare_training_artifact_paths,
    copy_shared_artifacts_to_training_subdir,
    get_seed_taxonomy_path,
    write_training_run_metadata,
)
from src.resume_state import default_training_state, load_json_state, now_timestamp, repair_training_state, save_json_state
from src.laminar_tracing import flush_laminar, init_laminar_from_env

def main():
    #* Configure
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description="Run AlphaOPT training")
    parser.add_argument(
        "--config",
        default=get_train_config_path(),
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resume and start from the configured entry point.",
    )
    args = parser.parse_args()

    os.environ["ALPHAOPT_TRAIN_CONFIG"] = args.config
    config = OmegaConf.load(args.config)

    from src.dataloader import DataLoader
    from src.utils import cal_time_cost
    from src.train_eval_utils import save_checkpoint, print_training_metrics_summary
    from library_online_learning import run_library_online_learning
    from library_diagnosis import run_library_diagnosis
    from library_refinement import run_library_refinement
    from src.experience_library import ExperienceLibrary
    from src.llm_programmer import ProgramGenerator
    from src.llm_diagnostic import ProgramDiagnostic
    from src.llm_extractor import InsightExtractor
    from src.llm_retriever import LibraryRetrieval
    from src.llm_evolver import LibraryEvolution

    config, run_lib_dir = prepare_training_artifact_paths(config, config_path=args.config)
    copied_shared_artifacts = copy_shared_artifacts_to_training_subdir(config)
    os.environ["ALPHAOPT_SEED_TAXONOMY_PATH"] = get_seed_taxonomy_path(config)
    OmegaConf.resolve(config)
    metadata_path = write_training_run_metadata(config, config_path=args.config)
    init_laminar_from_env(
        mode="training",
        config_path=args.config,
        metadata={
            "dataset": str(config.dataset),
            "output_folder": str(config.output_folder),
            "library_subdir": str(config.library_subdir),
            "base_model": str(config.base_model),
            "advanced_model": str(config.advanced_model),
            "base_service": str(config.base_service),
            "advanced_service": str(config.advanced_service),
        },
    )
    resume_enabled = bool(getattr(config, "resume", True)) and not args.no_resume
    resume_state_path = str(Path(config.file_paths.train_output_dir) / "training_resume_state.json")
    resume_state = default_training_state(config_path=args.config, library_subdir=str(config.library_subdir))
    if resume_enabled:
        resume_state = load_json_state(resume_state_path, resume_state)
    else:
        Path(resume_state_path).parent.mkdir(parents=True, exist_ok=True)
        save_json_state(resume_state_path, resume_state)
    print(f"Experience-library subdirectory: {config.library_subdir}")
    print(f"Experience-library artifacts: {run_lib_dir}")
    if copied_shared_artifacts:
        print(f"Copied shared artifacts: {', '.join(copied_shared_artifacts)}")
    print(f"Run metadata: {metadata_path}")

    # Initialize the LLM agents
    # Advanced models use OpenRouter
    llm_opt = ProgramGenerator(model=config.advanced_model, service=config.advanced_service, temperature=0)
    llm_diag = ProgramDiagnostic(model=config.advanced_model, service=config.advanced_service, temperature=0)
    llm_ins = InsightExtractor(model=config.advanced_model, service=config.advanced_service, temperature=0.7)

    temp_online = 0.7 if config.params.max_solution_attempts > 1 else 0
    llm_opt_online = ProgramGenerator(model=config.advanced_model, service=config.advanced_service, temperature=temp_online)

    # 0 (start from online learning), 1 (start from library diagnosis at iter 1)
    start_iter = int(config.start_iter or 0)
    end_iter = config.params.num_iterations + 1
    resume_phase_override = None
    active_iter = start_iter

    def _load_metrics_log():
        if os.path.exists(config.file_paths.metrics_log_path):
            with open(config.file_paths.metrics_log_path, "r") as f:
                return json.load(f)
        return []

    if resume_enabled and resume_state.get("status") in {"in_progress", "halted_transient_connection_error", "halted_provider_content_filter"}:
        current_phase = resume_state.get("current_phase")
        active_iter = int(resume_state.get("current_iter", start_iter) or start_iter)
        if current_phase == "online_learning":
            online_paths = (resume_state.get("online_learning") or {}).get("snapshot_paths", {})
            tasks_path = online_paths.get("tasks", f"{config.file_paths.train_output_dir}/train_tasks_record_base_snap.json")
            lib_path = online_paths.get("library", f"{config.file_paths.lib_dir}/library_base_snap.json")
            taxo_path = online_paths.get("taxonomy", f"{config.file_paths.lib_dir}/latest_taxonomy_base_snap.json")
            if all(os.path.exists(p) for p in (tasks_path, lib_path, taxo_path)):
                train_tasks = DataLoader(tasks_path, mode="learn", filter_success_num=None, reset=False)
                repaired = repair_training_state(
                    resume_state,
                    train_tasks,
                    enabled=bool(getattr(config, "repair", True)),
                    batch_size=int(config.params.batch_size),
                )
                if repaired:
                    print(f"Repaired training resume state; re-running {len(repaired)} task(s): {', '.join(repaired)}")
                    save_json_state(resume_state_path, resume_state)
                library = ExperienceLibrary.from_json_file(library_path=lib_path, taxonomy_path=taxo_path)
                metrics_log = _load_metrics_log()
                resume_phase_override = "online_learning"
                start_iter = 0
            else:
                print("Training resume snapshot for online learning is incomplete; starting from configured entry point.")
                resume_phase_override = None
        elif current_phase == "diagnosis":
            diag_paths = ((resume_state.get("diagnosis") or {}).get(str(active_iter)) or {}).get("snapshot_paths", {})
            tasks_path = diag_paths.get("tasks", f"{config.file_paths.train_output_dir}/train_tasks_record_iter{active_iter}_snap.json")
            lib_path = diag_paths.get("library", f"{config.file_paths.lib_dir}/library_iter{active_iter}_diag_snap.json")
            taxo_path = diag_paths.get("taxonomy", f"{config.file_paths.lib_dir}/latest_taxonomy_iter{active_iter}_snap.json")
            if all(os.path.exists(p) for p in (tasks_path, lib_path, taxo_path)):
                train_tasks = DataLoader(tasks_path, mode="learn", filter_success_num=None, reset=False)
                repaired = repair_training_state(
                    resume_state,
                    train_tasks,
                    enabled=bool(getattr(config, "repair", True)),
                    batch_size=int(config.params.batch_size),
                )
                if repaired:
                    print(f"Repaired training resume state; re-running {len(repaired)} task(s): {', '.join(repaired)}")
                    save_json_state(resume_state_path, resume_state)
                library = ExperienceLibrary.from_json_file(library_path=lib_path, taxonomy_path=taxo_path)
                metrics_log = _load_metrics_log()
                resume_phase_override = "diagnosis"
                start_iter = active_iter
            else:
                print("Training resume snapshot for diagnosis is incomplete; starting from configured entry point.")
                resume_phase_override = None
        elif current_phase == "refinement":
            refine_paths = ((resume_state.get("refinement") or {}).get(str(active_iter)) or {}).get("snapshot_paths", {})
            tasks_path = refine_paths.get("tasks", f"{config.file_paths.train_output_dir}/train_tasks_record_diag_iter{active_iter}.json")
            lib_path = refine_paths.get("library", f"{config.file_paths.lib_dir}/library_diag_iter{active_iter}.json")
            taxo_path = refine_paths.get("taxonomy", f"{config.file_paths.lib_dir}/latest_taxonomy_diag_iter{active_iter}.json")
            if all(os.path.exists(p) for p in (tasks_path, lib_path, taxo_path)):
                train_tasks = DataLoader(tasks_path, mode="learn", filter_success_num=None, reset=False)
                repaired = repair_training_state(
                    resume_state,
                    train_tasks,
                    enabled=bool(getattr(config, "repair", True)),
                    batch_size=int(config.params.batch_size),
                )
                if repaired:
                    print(f"Repaired training resume state; re-running {len(repaired)} task(s): {', '.join(repaired)}")
                    save_json_state(resume_state_path, resume_state)
                library = ExperienceLibrary.from_json_file(library_path=lib_path, taxonomy_path=taxo_path)
                metrics_log = _load_metrics_log()
                resume_phase_override = "refinement"
                start_iter = active_iter
            else:
                print("Training resume inputs for refinement are incomplete; starting from configured entry point.")
                resume_phase_override = None

    if "train_tasks" not in locals():
        if start_iter == 0:
            train_tasks = DataLoader(config.file_paths.train_data_path, mode="learn", filter_success_num=None, reset=True) 
            # Initialize the experience library as an empty list 
            library = ExperienceLibrary()
            # Track iteration metrics
            metrics_log = []  

        else:
            if start_iter == 1:
                train_data_path = f"{config.file_paths.train_output_dir}/train_tasks_record_base.json"
                lib_path = f"{config.file_paths.lib_dir}/library_base.json"
                taxo_path = f"{config.file_paths.lib_dir}/latest_taxonomy_base.json"
            else:
                train_data_path = f"{config.file_paths.train_output_dir}/train_tasks_record_diag_iter{start_iter-1}.json"
                lib_path = f"{config.file_paths.lib_dir}/library_refine_iter{start_iter-1}.json"
                taxo_path = f"{config.file_paths.lib_dir}/latest_taxonomy_diag_iter{start_iter-1}.json"
            train_tasks = DataLoader(train_data_path, mode="learn", filter_success_num=None, reset=False)
            library = ExperienceLibrary.from_json_file(
                            library_path = lib_path,
                            taxonomy_path = taxo_path)
            metrics_log = _load_metrics_log()

    # Run subset
    if config.data_slice:
        start = config.data_slice[0]
        end = config.data_slice[1]
        train_tasks = train_tasks.slice(start, end)

    start_time = time.time()
    resume_state["status"] = "in_progress"
    resume_state["updated_at"] = now_timestamp()
    save_json_state(resume_state_path, resume_state)
    for iter in range(start_iter, end_iter): 
        iter_start_time = time.time()
        # Update library retriever
        llm_retri = LibraryRetrieval(lib=library, model=config.base_model, service=config.base_service, temperature=0)

        #* Library online learning for once
        if iter == 0 and resume_phase_override in (None, "online_learning"):
            resume_state["current_phase"] = "online_learning"
            resume_state["current_iter"] = int(iter)
            resume_state["updated_at"] = now_timestamp()
            save_json_state(resume_state_path, resume_state)
            iter_metrics = run_library_online_learning(
                iter, 
                train_tasks, 
                llm_retri, llm_opt_online, llm_diag, llm_ins, library, 
                config.params,
                config.file_paths,
                resume_state=resume_state if resume_enabled else None,
                resume_state_path=resume_state_path if resume_enabled else None,
            )

            # Save checkpoint
            print(iter_metrics)
            metrics_log.append(iter_metrics)
            save_checkpoint(library=library, tasks=train_tasks, metrics=metrics_log, paths=config.file_paths, suffix="base")
            resume_state["current_phase"] = "diagnosis"
            resume_state["current_iter"] = 1
            resume_state["updated_at"] = now_timestamp()
            save_json_state(resume_state_path, resume_state)
            # directly continue to iter 1
            resume_phase_override = None
            continue

        #* Library Diagnosis
        if resume_phase_override in (None, "diagnosis"):
            resume_state["current_phase"] = "diagnosis"
            resume_state["current_iter"] = int(iter)
            resume_state["updated_at"] = now_timestamp()
            save_json_state(resume_state_path, resume_state)
            iter_metrics = run_library_diagnosis(
                iter, 
                train_tasks, 
                llm_retri, llm_opt, llm_diag, llm_ins, library, 
                config.params,
                config.file_paths,
                max_workers=12,
                resume_state=resume_state if resume_enabled else None,
                resume_state_path=resume_state_path if resume_enabled else None,
            )
            
            # Save checkpoint
            print(iter_metrics)
            if len(metrics_log) > iter:
                metrics_log[iter] = iter_metrics
            else:
                metrics_log.append(iter_metrics)
            save_checkpoint(library=library, tasks=train_tasks, metrics=metrics_log, paths=config.file_paths, suffix=f"diag_iter{iter}")
        else:
            iter_metrics = metrics_log[iter] if len(metrics_log) > iter else {}

        # #* Library Refinement
        resume_state["current_phase"] = "refinement"
        resume_state["current_iter"] = int(iter)
        resume_state["updated_at"] = now_timestamp()
        save_json_state(resume_state_path, resume_state)
        llm_evolve = LibraryEvolution(lib=library, model=config.base_model, service=config.base_service, temperature=0.7)
        (
            refined_library,
            avg_refinement_rate,
            token_usage_delta,
            duration_min,
            refined_ins_num,
            refinement_success_rate,
            refinement_success_num,
        ) = run_library_refinement(
            iter=iter, tasks=train_tasks, 
            config=config, llm_evolve=llm_evolve,
            verbose=False, save_data=True, output_path=config.file_paths.train_output_dir,
            max_workers=8,
            resume_state=resume_state if resume_enabled else None,
            resume_state_path=resume_state_path if resume_enabled else None,
        )
        print("refinement_avg_gain:", avg_refinement_rate)
        # Save iteration metrics log for library evolution phase
        last_metrics = metrics_log[-1]
        last_metrics["refinement_avg_gain"] = round(avg_refinement_rate, 3)
        last_metrics["refined_ins_num"] = int(refined_ins_num)
        # Requested metrics: accepted refinements / refinements attempted
        last_metrics["refinement_success_rate"] = round(float(refinement_success_rate), 3)
        last_metrics["refinement_success_num"] = int(refinement_success_num)
        last_metrics["refinement_proposed_num"] = int(refined_ins_num)
        last_metrics["refinement_token_usage"] = token_usage_delta
        last_metrics["library_refinement_duration (min)"] = round(float(duration_min), 3)


        # #* The chosen best variant for the next round
        library = refined_library
        # Save library
        save_checkpoint(library=library, tasks=None, metrics=metrics_log, paths=config.file_paths, suffix=f"refine_iter{iter}")
        resume_state["refinement"][str(iter)] = {
            "status": "completed",
            "completed_at": now_timestamp(),
        }
        resume_state["current_phase"] = "diagnosis"
        resume_state["current_iter"] = int(iter + 1)
        resume_state["updated_at"] = now_timestamp()
        save_json_state(resume_state_path, resume_state)
        resume_phase_override = None

        iter_duration = cal_time_cost(iter_start_time, f'Iteration {iter} Total Pipeline')

    # Count time cost
    total_duration = cal_time_cost(start_time, f'The iterative library learning and evolution process for {config.params.num_iterations} iterations')

    # Print structured metrics summary
    print_training_metrics_summary(metrics_log)
    resume_state["status"] = "completed"
    resume_state["current_phase"] = None
    resume_state["updated_at"] = now_timestamp()
    save_json_state(resume_state_path, resume_state)
    flush_laminar()


if __name__ == "__main__":
    main()
