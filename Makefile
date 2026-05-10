.PHONY: help sync sync-extras smoke train eval check-proxy train-with-proxy exp-e1 exp-e2 ensure-e1-library ensure-e2-library

UV ?= uv
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(REPO_ROOT:/=)
TRAIN_CONFIG ?= configs/train/default.yaml
EVAL_CONFIG ?= configs/eval/default.yaml
E2_TRAIN_CONFIG := configs/train/gpt55_gurobi_train_data_all_452.yaml
E1_EVAL_CONFIG := configs/eval/gpt55_original_library.yaml
E2_EVAL_CONFIG := configs/eval/gpt55_retrained_library.yaml
E1_LIBRARY := data/experience_library/default/library.json
E1_TAXONOMY := data/experience_library/default/latest_taxonomy_new.json
E2_ARTIFACT_GLOB := data/experience_library/gpt55_gurobi_train_data_all_452*

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' "$(REPO_ROOT)/Makefile" | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

sync: ## Install locked dependencies (uv sync)
	cd "$(REPO_ROOT)" && $(UV) sync

sync-extras: ## Install with optional extras (notebooks, retrieval, alternate solvers)
	cd "$(REPO_ROOT)" && $(UV) sync --all-extras

smoke: ## Quick import check for core modules
	cd "$(REPO_ROOT)" && $(UV) run python -c "import gurobipy; import src.utils; import src.train_eval_utils; print('ok')"

train: ## Run training with TRAIN_CONFIG=... (default: configs/train/default.yaml)
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python main.py --config "$(TRAIN_CONFIG)"

eval: ## Run evaluation with EVAL_CONFIG=... and TRAIN_CONFIG=... for pricing/API context
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(EVAL_CONFIG)"

check-proxy: ## Verify OpenAI-compatible proxy at localhost:8801 (optional local setup)
	@curl -sf http://localhost:8801/health >/dev/null && echo "proxy health OK" || (echo "debug: no proxy at http://localhost:8801/health — use direct API keys or set base_service to your proxy URL in configs/train/default.yaml" >&2; exit 1)

train-with-proxy: check-proxy ## Run training only after proxy health check passes
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python main.py --config "$(TRAIN_CONFIG)"

ensure-e1-library:
	@test -f "$(REPO_ROOT)/$(E1_LIBRARY)" || (echo "missing E1 library: $(E1_LIBRARY)" >&2; exit 1)
	@test -f "$(REPO_ROOT)/$(E1_TAXONOMY)" || (echo "missing E1 taxonomy: $(E1_TAXONOMY)" >&2; exit 1)

ensure-e2-library:
	@artifact_dir=$$(find "$(REPO_ROOT)/data/experience_library" -maxdepth 1 -mindepth 1 -type d -name 'gpt55_gurobi_train_data_all_452*' -print0 | xargs -0 -r ls -dt 2>/dev/null | head -n 1); \
	test -n "$$artifact_dir" || (echo "missing E2 artifact directory. Run 'make exp-e2' training step first." >&2; exit 1); \
	test -f "$$artifact_dir/library_refine_iter1.json" || (echo "missing E2 library: $$artifact_dir/library_refine_iter1.json. Run 'make exp-e2' training step first." >&2; exit 1); \
	test -f "$$artifact_dir/latest_taxonomy_refine_iter1.json" || (echo "missing E2 taxonomy: $$artifact_dir/latest_taxonomy_refine_iter1.json. Run 'make exp-e2' training step first." >&2; exit 1)

exp-e1: ensure-e1-library ## E1: evaluate GPT-5.5 on all eval datasets with the preserved default library
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(E1_EVAL_CONFIG)"

exp-e2: ## E2: retrain library with GPT-5.5+Gurobi on original train set, then evaluate on all eval datasets
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python main.py --config "$(E2_TRAIN_CONFIG)"
	@artifact_dir=$$(find "$(REPO_ROOT)/data/experience_library" -maxdepth 1 -mindepth 1 -type d -name 'gpt55_gurobi_train_data_all_452*' -print0 | xargs -0 -r ls -dt 2>/dev/null | head -n 1); \
	test -n "$$artifact_dir" || (echo "training finished but no E2 artifact directory was found." >&2; exit 1); \
	test -f "$$artifact_dir/library_refine_iter1.json" || (echo "training finished but expected E2 library was not produced: $$artifact_dir/library_refine_iter1.json" >&2; exit 1); \
	test -f "$$artifact_dir/latest_taxonomy_refine_iter1.json" || (echo "training finished but expected E2 taxonomy was not produced: $$artifact_dir/latest_taxonomy_refine_iter1.json" >&2; exit 1); \
	tmp_cfg=$$(mktemp); \
	sed -e "s#^library_path: .*#library_path: ./data/experience_library/$$(basename "$$artifact_dir")/library_refine_iter1.json#" \
	    -e "s#^taxonomy_path: .*#taxonomy_path: ./data/experience_library/$$(basename "$$artifact_dir")/latest_taxonomy_refine_iter1.json#" \
	    "$(REPO_ROOT)/$(E2_EVAL_CONFIG)" > "$$tmp_cfg"; \
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$$tmp_cfg"; \
	rm -f "$$tmp_cfg"
