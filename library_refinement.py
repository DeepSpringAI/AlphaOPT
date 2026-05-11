import json
import traceback
import copy
import time
import os
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import LLMContentFilterError, LLMTransientError, save_log_data, get_token_usage
from src.experience_library import ExperienceLibrary
from src.dataloader import DataLoader 
from src.llm_retriever import LibraryRetrieval
from src.llm_evolver import LibraryEvolution
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
from src.resume_state import make_refinement_result, mark_training_halted, now_timestamp, save_json_state, task_key

from src.prompts.prompts_evolve import PROMPT_INS_REFINEMENT


def run_library_refinement(
    iter,
    tasks,
    config,
    llm_evolve,
    verbose=False,
    save_data=False,
    output_path=None,
    max_workers=4,
    resume_state=None,
    resume_state_path: str | None = None,
):
    """
    Parallelize only the outer loop over insights.
    """
    _start_time = time.time()

    def _process_one_insight(ins):
        """
        Process a single insight:
            1) Collect neg/unr reasons, call LLM to generate conditions per task.
            2) Integrate into a refinement prompt, get K new conditions.
            3) Build K library variants and verify retrieval to choose the best.
        Returns a dict with final condition and distribution to be merged in the main thread.
        """
        dataset_name = getattr(config, "dataset", None) or ",".join(map(str, getattr(config, "datasets", []) or []))
        with trace_span(
            "alphaopt.task",
            input={"insight_id": getattr(ins, "insight_id", None), "condition": getattr(ins, "condition", None)},
            tags=["alphaopt", "train", "refinement", f"iter-{iter}", f"dataset:{dataset_name}" if dataset_name else ""],
            metadata={
                "mode": "training",
                "phase": "refinement",
                "dataset": dataset_name,
                "insight_id": getattr(ins, "insight_id", None),
                "iteration": iter,
                "output_path": output_path,
            },
            attributes={
                "alphaopt.mode": "training",
                "alphaopt.phase": "refinement",
                "alphaopt.dataset": dataset_name,
                "alphaopt.insight_id": str(getattr(ins, "insight_id", "")),
                "alphaopt.iteration": int(iter),
            },
            session_id=f"refinement:{config.output_folder}:iter-{iter}",
        ) as root_span:
          try:
            set_trace_metadata(
                {
                    "mode": "training",
                    "phase": "refinement",
                    "dataset": dataset_name,
                    "insight_id": getattr(ins, "insight_id", None),
                    "iteration": iter,
                    "output_path": output_path,
                }
            )
            res = _process_one_insight_impl(ins)
            trace_id = current_trace_id()
            status = "skipped" if not res else ("accepted" if res.get("refinement_accepted") else "not_accepted")
            set_span_attributes(root_span, {"alphaopt.status": status, "alphaopt.laminar_trace_id": trace_id})
            add_span_tags(root_span, [status])
            set_span_output(root_span, res or {"status": status})
            record_trace_index(
                output_path,
                {
                    "mode": "training",
                    "phase": "refinement",
                    "task_id": "",
                    "iteration": iter,
                    "insight_id": getattr(ins, "insight_id", None),
                    "trace_id": trace_id,
                    "status": status,
                    "artifact_path": output_path,
                },
            )
            return res
          except (LLMContentFilterError, LLMTransientError) as exc:
            trace_id = current_trace_id()
            status = "provider_policy_blocked" if isinstance(exc, LLMContentFilterError) else "transient_connection_halt"
            laminar_record_exception(root_span, exc)
            add_span_tags(root_span, [status])
            set_span_attributes(root_span, {"alphaopt.status": status, "alphaopt.laminar_trace_id": trace_id})
            set_span_output(root_span, {"status": status, "error": str(exc)[:1000]})
            record_trace_index(
                output_path,
                {
                    "mode": "training",
                    "phase": "refinement",
                    "task_id": "",
                    "iteration": iter,
                    "insight_id": getattr(ins, "insight_id", None),
                    "trace_id": trace_id,
                    "status": status,
                    "artifact_path": output_path,
                },
            )
            raise
          except Exception as exc:
            trace_id = current_trace_id()
            laminar_record_exception(root_span, exc)
            add_span_tags(root_span, ["experiment-error"])
            set_span_attributes(root_span, {"alphaopt.status": "experiment_error", "alphaopt.laminar_trace_id": trace_id})
            record_trace_index(
                output_path,
                {
                    "mode": "training",
                    "phase": "refinement",
                    "task_id": "",
                    "iteration": iter,
                    "insight_id": getattr(ins, "insight_id", None),
                    "trace_id": trace_id,
                    "status": "experiment_error",
                    "artifact_path": output_path,
                },
            )
            raise

    def _process_one_insight_impl(ins):
        """
        Process a single insight implementation. Wrapped by _process_one_insight
        to provide one Laminar root trace per refinement candidate.
        """

        # Copy lists to avoid accidental in-place mutation on shared objects
        pos_task_ids = list(ins.distribution.get("positive") or [])
        neg_task_ids = list(ins.distribution.get("negative") or [])
        unr_task_ids = list(ins.distribution.get("unretrieved") or [])
        guard_unr_task_ids = list(ins.distribution.get("guard_unretrieved") or [])
        guard_neg_task_ids = list(ins.distribution.get("guard_negative") or [])
        # Tasks that were historically labeled "positive" but later became NOT retrieved after refinement.
        # We keep them for evaluation to track/penalize positive regressions across iterations.
        guard_lost_pos_task_ids = list(ins.distribution.get("guard_lost_positive") or [])

        def _dedup_keep_order(xs):
            seen = set()
            out = []
            for x in xs:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        # If there are neither negative nor unretrieved tasks, skip refinement
        if not neg_task_ids and not unr_task_ids:
            return None

        # Add successful tasks that actually retrieved this insight into positive
        for task in tasks:
            if task.output_status[-1] == "optimal":
                if ins.insight_id in task.retri_ins_lst:
                    pos_task_ids.append(task.id)
        pos_task_ids = _dedup_keep_order(pos_task_ids)

        # Generate conditions for negative tasks
        neg_condition_lst = []
        for task in tasks.subset_by_ids(neg_task_ids):
            neg_condition = llm_evolve.generate_neg_condition(task, ins, iter, verbose=verbose, output_dir=output_path)
            neg_condition_lst.append(neg_condition)

        # Generate conditions for unretrieved tasks
        unr_condition_lst = []
        for task in tasks.subset_by_ids(unr_task_ids):
            unr_condition = llm_evolve.generate_unr_condition(task, ins, iter, verbose=verbose, output_dir=output_path)
            unr_condition_lst.append(unr_condition)

        #* Refine insight conditions 
        refined_conditions_k = llm_evolve.refine_insight(
            iter,
            neg_condition_lst,
            unr_condition_lst,
            ins,
            config.params.variant_num,
            verbose=verbose,
            output_dir=output_path,
        )

        # Build K library variants and evaluate
        library_variants_k = llm_evolve.build_library_variant(ins.insight_id, refined_conditions_k)

        # Evaluation task sets
        # Include guard sets in evaluation to avoid accepting changes that regress previously-fixed tasks.
        eval_pos_task_ids = _dedup_keep_order(pos_task_ids + guard_unr_task_ids + guard_lost_pos_task_ids)
        eval_neg_task_ids = _dedup_keep_order(neg_task_ids + guard_neg_task_ids)
        eval_unr_task_ids = _dedup_keep_order(unr_task_ids)

        total_tasks_num = len(eval_pos_task_ids + eval_neg_task_ids + eval_unr_task_ids)

        # Baseline performance BEFORE refinement.
        base_pos_retri_count = len(eval_pos_task_ids)
        base_neg_retri_count = len(_dedup_keep_order(neg_task_ids))
        base_unr_retri_count = 0

        base_performance = (base_pos_retri_count + base_unr_retri_count + len(eval_neg_task_ids) - base_neg_retri_count) / total_tasks_num if total_tasks_num > 0 else 0

        best_performance = base_performance
        best_pos_retri_count = base_pos_retri_count
        best_neg_retri_count = base_neg_retri_count
        best_unr_retri_count = base_unr_retri_count
        # Baseline "matched" sets derived from the same assumptions as base_*_retri_count.
        # This prevents accidentally treating everything as solved when no variant beats the baseline.
        best_matched_pos_tids = list(eval_pos_task_ids)  # assume all should-retrieve tasks are retrieved at baseline
        best_matched_neg_tids = list(_dedup_keep_order(neg_task_ids))  # assume active negatives are retrieved at baseline
        best_matched_unr_tids = []  # assume unretrieved tasks are NOT retrieved at baseline
        latest_condition = getattr(ins, "condition", None)

        # Decide which retrieval stage to use for this insight.
        # If the insight is a Code Implementation insight, it should be retrieved in the Program stage,
        # which requires a formulation (mathematical model) as context.
        ins_taxo = getattr(ins, "taxonomy", None) or {}
        ins_stage = "Program" if "Code Implementation" in ins_taxo else "Formulation"

        # For Program-stage verification, load the formulation from the previous round files:
        # learning/{output_folder}/task_{ID}/model_iter_{i}.txt
        # Cache per task-id to avoid repeated disk reads during variant evaluation.
        _formulation_cache = {}

        def _load_formulation_for_task(task):
            if task.id in _formulation_cache:
                return _formulation_cache[task.id]
            if not output_path:
                _formulation_cache[task.id] = None
                return None
            fp = os.path.join(str(output_path), f"task_{task.id}", f"model_iter_{iter}.txt")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    txt = f.read()
                txt = txt.strip() if isinstance(txt, str) else txt
                _formulation_cache[task.id] = txt
                return txt
            except Exception:
                _formulation_cache[task.id] = None
                return None

        formulation_lookup = _load_formulation_for_task if ins_stage == "Program" else None

        # Evaluate each variant
        for i, lib in enumerate(library_variants_k):
            llm_retri = LibraryRetrieval(lib=lib, model=config.base_model, service=config.base_service, temperature=0)
            pos_retri_count, matched_pos_tids = llm_evolve.verify_retrieval(
                ins.insight_id, tasks, eval_pos_task_ids, llm_retri,
                stage=ins_stage, config_override=config, formulation_lookup=formulation_lookup
            )
            neg_retri_count, matched_neg_tids = llm_evolve.verify_retrieval(
                ins.insight_id, tasks, eval_neg_task_ids, llm_retri,
                stage=ins_stage, config_override=config, formulation_lookup=formulation_lookup
            )
            # unr_retri_count means the number of tasks that have been retrieved
            unr_retri_count, matched_unr_tids = llm_evolve.verify_retrieval(
                ins.insight_id, tasks, eval_unr_task_ids, llm_retri,
                stage=ins_stage, config_override=config, formulation_lookup=formulation_lookup
            )

            # Variant scoring metric: the number of retrieved pos, unr insights and the decrease in neg insights number
            variant_performance = (pos_retri_count + unr_retri_count + len(eval_neg_task_ids) - neg_retri_count) / total_tasks_num if total_tasks_num > 0 else 0

            if variant_performance > best_performance:
                best_performance = variant_performance
                latest_condition = refined_conditions_k[i] if i < len(refined_conditions_k) else latest_condition
                best_pos_retri_count = pos_retri_count
                best_neg_retri_count = neg_retri_count
                best_unr_retri_count = unr_retri_count
                best_matched_pos_tids = matched_pos_tids
                best_matched_neg_tids = matched_neg_tids
                best_matched_unr_tids = matched_unr_tids

        performance_gain = best_performance - base_performance
        refinement_accepted = performance_gain > 0
        
        #* Update ins.distribution based on best variant results (new)
        # Remove solved active-negative tasks (no longer retrieved), but keep them in guard_negative
        # so that future refinements are evaluated against them (regression protection).
        solved_neg_tids = set(neg_task_ids) - (set(best_matched_neg_tids) & set(neg_task_ids))
        if solved_neg_tids:
            ins.distribution["negative"] = [
                tid for tid in ins.distribution.get("negative", [])
                if tid not in solved_neg_tids
            ]
            ins.distribution.setdefault("guard_negative", [])
            for tid in sorted(solved_neg_tids):
                if tid not in ins.distribution["guard_negative"]:
                    ins.distribution["guard_negative"].append(tid)
        
        # Remove solved active-unretrieved tasks (now retrieved), but keep them in guard_unretrieved
        # so that future refinements are evaluated against them (regression protection).
        solved_unr_tids = (set(best_matched_unr_tids) & set(unr_task_ids))
        if solved_unr_tids:
            ins.distribution["unretrieved"] = [
                tid for tid in ins.distribution.get("unretrieved", [])
                if tid not in solved_unr_tids
            ]
            # Store as "guard_unretrieved": previously unretrieved but fixed (now retrieved).
            ins.distribution.setdefault("guard_unretrieved", [])
            for tid in sorted(solved_unr_tids):
                if tid not in ins.distribution["guard_unretrieved"]:
                    ins.distribution["guard_unretrieved"].append(tid)

        # Move positive tasks that are no longer retrieved after refinement into guard_lost_positive.
        # This tracks positive regressions across iterations without mixing them into other buckets.
        active_pos_tids = set(ins.distribution.get("positive") or [])
        matched_active_pos_tids = set(best_matched_pos_tids) & active_pos_tids
        lost_pos_tids = active_pos_tids - matched_active_pos_tids
        if lost_pos_tids:
            ins.distribution["positive"] = [
                tid for tid in ins.distribution.get("positive", [])
                if tid not in lost_pos_tids
            ]
            ins.distribution.setdefault("guard_lost_positive", [])
            for tid in sorted(lost_pos_tids):
                if tid not in ins.distribution["guard_lost_positive"]:
                    ins.distribution["guard_lost_positive"].append(tid)
        
        # Return a compact result to be merged by the main thread
        return {
            "insight_id": ins.insight_id,
            "orig_condition": getattr(ins, "condition", None),
            "latest_condition": latest_condition,
            "distributions": {
                "positive": best_matched_pos_tids,
                "negative": best_matched_neg_tids,
                "unretrieved": best_matched_unr_tids
                },
            "performance_gain": performance_gain,
            "refinement_accepted": refinement_accepted,
            "report": (
                f"\nBest Performance on insight {ins.insight_id}: {best_performance} "
                f"\n Performance Gain: {performance_gain}"
                f"\npositive (before: {len(pos_task_ids)}; after: {best_pos_retri_count}) "
                f"\nnegative (before: {len(neg_task_ids)}; after: {best_neg_retri_count}) "
                f"\nunretrieved (before: {len(unr_task_ids)}; after: {len(unr_task_ids) - best_unr_retri_count})"
            )
        }

    # Results dictionaries (updated only in the main thread; no locks needed)
    refined_insights = {}         # {insight_id: [original_condition, refined_condition]}
    insight_distributions = {}    # {insight_id: {"positive": [...], "negative": [...], "unretrieved": [...]}}
    refinement_state = {}
    completed_insight_ids = set()
    insight_results = {}
    if resume_state is not None:
        refinement_state = resume_state.setdefault("refinement", {}).setdefault(
            str(iter),
            {
                "status": "in_progress",
                "completed_insight_ids": [],
                "insight_results": {},
            },
        )
        completed_insight_ids = {task_key(insight_id) for insight_id in refinement_state.get("completed_insight_ids", [])}
        insight_results = refinement_state.setdefault("insight_results", {})
        for key, record in insight_results.items():
            result = (record or {}).get("result") or {}
            status = result.get("status")
            if status not in {"accepted", "not_accepted", "provider_policy_blocked"}:
                continue
            result_iid = result.get("insight_id", key)
            if "orig_condition" in result and "latest_condition" in result:
                refined_insights[result_iid] = [result.get("orig_condition"), result.get("latest_condition")]
            if isinstance(result.get("distributions"), dict):
                insight_distributions[result_iid] = result["distributions"]

    # Preselect insights to run for proper tqdm progress
    candidate_insights = [ins for ins in llm_evolve.library]
    total = len(candidate_insights)
    refined_ins_num = 0
    refinement_success_num = 0
    total_performance_gain = 0 
    for record in insight_results.values():
        result = (record or {}).get("result") or {}
        status = result.get("status")
        if status in {"accepted", "not_accepted", "provider_policy_blocked"}:
            refined_ins_num += 1
            total_performance_gain += float(result.get("performance_gain", 0) or 0)
        if status == "accepted":
            refinement_success_num += 1

    # Track token usage before refinement
    usage_before = get_token_usage()

    def _write_refinement_partial_artifacts() -> None:
        if save_data and output_path:
            refined_ins_list = [
                {
                    "insight_id": insight_id,
                    "original_condition": conds[0],
                    "refined_condition": conds[1]
                }
                for insight_id, conds in refined_insights.items()
            ]
            save_log_data(refined_ins_list, f"{output_path}/refined_insights_iter{iter}_partial.json")
            save_log_data(insight_distributions, f"{output_path}/refined_insight_distributions_iter{iter}_partial.json")
        if resume_state is not None and resume_state_path:
            refinement_state["snapshot_paths"] = {
                "library": f"{config.file_paths.lib_dir}/library_diag_iter{iter}.json",
                "taxonomy": f"{config.file_paths.lib_dir}/latest_taxonomy_diag_iter{iter}.json",
                "tasks": f"{config.file_paths.train_output_dir}/train_tasks_record_diag_iter{iter}.json",
                "partial_refined_insights": f"{output_path}/refined_insights_iter{iter}_partial.json" if output_path else None,
                "partial_refined_distributions": f"{output_path}/refined_insight_distributions_iter{iter}_partial.json" if output_path else None,
            }
            refinement_state["completed_insight_ids"] = sorted(completed_insight_ids)
            refinement_state["insight_results"] = insight_results
            resume_state["current_phase"] = "refinement"
            resume_state["current_iter"] = int(iter)
            resume_state["updated_at"] = now_timestamp()
            save_json_state(resume_state_path, resume_state)

    # Thread pool over insights only
    executor = None
    try:
      executor = ThreadPoolExecutor(max_workers=max_workers)
      with tqdm(total=total, initial=len(completed_insight_ids), desc=f"[Iteration {iter}] Library Refinement") as pbar:
        future_map = {
            executor.submit(_process_one_insight, ins): ins.insight_id
            for ins in candidate_insights
            if task_key(ins.insight_id) not in completed_insight_ids
        }
        for fut in as_completed(future_map):
            iid = future_map[fut]
            try:
                res = fut.result()
            except LLMContentFilterError as exc:
                print(f"\n   [WARNING] Insight {iid}: provider policy blocked refinement; recording blocked insight and continuing.\n")
                res = {
                    "insight_id": iid,
                    "orig_condition": None,
                    "latest_condition": None,
                    "distributions": {},
                    "performance_gain": 0,
                    "refinement_accepted": False,
                    "provider_policy_blocked": True,
                    "report": f"\nProvider policy blocked refinement for insight {iid}; skipped mutation.",
                    "error": str(exc)[:4000],
                }
                record_trace_index(
                    output_path,
                    {
                        "mode": "training",
                        "phase": "refinement",
                        "iteration": iter,
                        "insight_id": iid,
                        "trace_id": current_trace_id(),
                        "status": "provider_policy_blocked",
                        "artifact_path": output_path,
                    },
                )
            except LLMTransientError as exc:
                for pending_future in future_map:
                    if pending_future is not fut:
                        pending_future.cancel()
                if resume_state is not None and resume_state_path:
                    refinement_state["status"] = "halted_transient_connection_error"
                    mark_training_halted(
                        resume_state,
                        phase="refinement",
                        iter=iter,
                        status="halted_transient_connection_error",
                        unit_id=iid,
                        error=exc,
                    )
                    _write_refinement_partial_artifacts()
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = None
                raise
            except Exception:
                traceback.print_exc()
                pbar.update(1)
                continue
            # The insight do not have negatvie or unretrieved tasks
            if not res:
                completed_insight_ids.add(task_key(iid))
                if resume_state is not None and resume_state_path:
                    insight_results[task_key(iid)] = make_refinement_result(
                        insight_id=iid,
                        status="skipped",
                        result={"status": "skipped"},
                    )
                    _write_refinement_partial_artifacts()
                pbar.update(1)
                continue

            refined_ins_num += 1 
            total_performance_gain += res["performance_gain"]
            if res.get("refinement_accepted"):
                refinement_success_num += 1
            iid = res["insight_id"]
            status = "provider_policy_blocked" if res.get("provider_policy_blocked") else ("accepted" if res.get("refinement_accepted") else "not_accepted")
            if not res.get("provider_policy_blocked"):
                refined_insights[iid] = [res["orig_condition"], res["latest_condition"]]
                insight_distributions[iid] = res["distributions"]
            completed_insight_ids.add(task_key(iid))
            if resume_state is not None and resume_state_path:
                insight_results[task_key(iid)] = make_refinement_result(
                    insight_id=iid,
                    status=status,
                    result={**res, "status": status},
                    error=res.get("error"),
                )
                _write_refinement_partial_artifacts()
            # Print once per completed insight to avoid interleaved outputs from threads
            print(res["report"])
            pbar.update(1)
    finally:
      if executor is not None:
        executor.shutdown(wait=True)

    # persist results
    if save_data and output_path:
        refined_ins_list = [
            {
                "insight_id": insight_id,
                "original_condition": conds[0],
                "refined_condition": conds[1]
            }
            for insight_id, conds in refined_insights.items()
        ]
        save_log_data(refined_ins_list, f"{output_path}/refined_insights_iter{iter}.json")
        save_log_data(insight_distributions, f"{output_path}/refined_insight_distributions_iter{iter}.json")
    if resume_state is not None and resume_state_path:
        refinement_state["status"] = "completed"
        refinement_state["completed_insight_ids"] = sorted(completed_insight_ids)
        refinement_state["insight_results"] = insight_results
        refinement_state["snapshot_paths"] = {
            "library": f"{config.file_paths.lib_dir}/library_diag_iter{iter}.json",
            "taxonomy": f"{config.file_paths.lib_dir}/latest_taxonomy_diag_iter{iter}.json",
            "tasks": f"{config.file_paths.train_output_dir}/train_tasks_record_diag_iter{iter}.json",
            "refined_insights": f"{output_path}/refined_insights_iter{iter}.json" if output_path else None,
            "refined_distributions": f"{output_path}/refined_insight_distributions_iter{iter}.json" if output_path else None,
        }
        resume_state["updated_at"] = now_timestamp()
        save_json_state(resume_state_path, resume_state)
    
    # Calculate the average refinement again (the average proportion of solved retrieval-misaligned tasks per insight)
    avg_refinement_rate = total_performance_gain / refined_ins_num if refined_ins_num else 0

    # Token usage summary for this phase
    usage_after = get_token_usage()
    token_usage_delta = {}
    for vendor, stats_after in usage_after.items():
        stats_before = usage_before.get(vendor, {})
        vendor_delta = {
            k: float(stats_after.get(k, 0.0) - stats_before.get(k, 0.0))
            for k in ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "cost")
        }
        # Keep vendors with any activity. Cost can be zero when pricing is unknown,
        # but request/token counts are still needed for audit and price debugging.
        if any(float(vendor_delta.get(k, 0.0) or 0.0) != 0.0 for k in ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "cost")):
            token_usage_delta[vendor] = vendor_delta

    # Write refined conditions back to a copied library
    refined_library = copy.deepcopy(llm_evolve.library)
    for ins in refined_library:
        if ins.insight_id in refined_insights:
            ins.condition = refined_insights[ins.insight_id][1]
            # Increment refine_version for successfully refined insights
            ins.refine_version += 1
    # Attach token usage so caller can log if needed
    duration_min = (time.time() - _start_time) / 60.0
    # Additional refinement success metrics:
    # refined_ins_num: number of insights that needed refinement (had negative/unretrieved tasks)
    # refinement_success_num: number of insights whose refined variant was accepted (beat baseline)
    refinement_success_rate = (refinement_success_num / refined_ins_num) if refined_ins_num else 0
    return (
        refined_library,
        avg_refinement_rate,
        token_usage_delta,
        duration_min,
        refined_ins_num,
        refinement_success_rate,
        refinement_success_num,
    )


# Test a demo
if __name__ == "__main__":
    import time
    from datetime import datetime
    from src.utils import cal_time_cost
    from src.train_eval_utils import save_checkpoint

    #* Configure
    from omegaconf import OmegaConf
    from src.config import load_train_config

    config = load_train_config()
    init_laminar_from_env(
        mode="training",
        config_path=os.getenv("ALPHAOPT_TRAIN_CONFIG"),
        metadata={"phase": "refinement"},
    )

    start_time = time.time()

    # Load previous library
    # 0 (start from online learning), 1 (start from library diagnosis at iter 1)
    start_iter = config.start_iter 
    library_path = f"{config.file_paths.lib_dir}/library_diag_iter{start_iter}.json"
    taxo_path = f"{config.file_paths.lib_dir}/latest_taxonomy_diag_iter{start_iter}.json"
    library = ExperienceLibrary.from_json_file(
                                library_path = library_path,
                                taxonomy_path = taxo_path)

    # Load training data
    task_path = f"{config.file_paths.train_output_dir}/train_tasks_record_diag_iter{start_iter}.json"
    train_tasks = DataLoader(task_path, mode="learn")

    # Library Evoluation
    llm_evolve = LibraryEvolution(lib=library, model=config.advanced_model, service=config.advanced_service, temperature=0.7)
    (
        refined_library,
        avg_refinement_rate,
        token_usage_delta,
        duration_min,
        refined_ins_num,
        refinement_success_rate,
        refinement_success_num,
    ) = run_library_refinement(
        iter=start_iter, tasks=train_tasks, 
        config=config, llm_evolve=llm_evolve,
        verbose=False, save_data=True, output_path=config.file_paths.train_output_dir,
        max_workers=8,
    )

    # Track iteration metrics
    with open(config.file_paths.metrics_log_path, "r") as f:
        metrics_log = json.load(f)

    last_metrics = metrics_log[-1]
    last_metrics["refinement_avg_gain"] = round(avg_refinement_rate, 3)
    last_metrics["refined_ins_num"] = int(refined_ins_num)
    # Requested metrics: accepted refinements / refinements attempted
    last_metrics["refinement_success_rate"] = round(float(refinement_success_rate), 3)
    last_metrics["refinement_success_num"] = int(refinement_success_num)
    last_metrics["refinement_proposed_num"] = int(refined_ins_num)
    last_metrics["refinement_token_usage"] = token_usage_delta
    last_metrics["library_refinement_duration (min)"] = round(float(duration_min), 3)

    
    print("refinement_avg_gain:", avg_refinement_rate)
    # Save library
    save_checkpoint(library=refined_library, tasks=None, metrics=metrics_log, paths=config.file_paths, suffix=f"refine_iter{start_iter}")

    # Count time cost
    total_duration = cal_time_cost(start_time, f'The library refinement process')
    flush_laminar()
