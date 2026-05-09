.PHONY: help sync sync-extras smoke train eval check-proxy train-with-proxy exp-e1 exp-e2 ensure-e1-library ensure-e2-library

UV ?= uv
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(REPO_ROOT:/=)
TRAIN_CONFIG ?= train_config.yaml
EVAL_CONFIG ?= eval_config.yaml
E2_TRAIN_CONFIG := train_config_e2_gpt54_gurobi.yaml
E1_EVAL_CONFIG := eval_config_e1_gpt54_original.yaml
E2_EVAL_CONFIG := eval_config_e2_gpt54_retrained.yaml
E1_LIBRARY := data/experience_library/original_gpt4o_gurobi_library.json
E1_TAXONOMY := data/experience_library/original_gpt4o_gurobi_taxonomy.json
E2_LIBRARY := data/experience_library/iterations/e2_gpt54_gurobi_train_data_all_452/library_refine_iter1.json
E2_TAXONOMY := data/experience_library/iterations/e2_gpt54_gurobi_train_data_all_452/latest_taxonomy_refine_iter1.json

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' "$(REPO_ROOT)/Makefile" | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

sync: ## Install locked dependencies (uv sync)
	cd "$(REPO_ROOT)" && $(UV) sync

sync-extras: ## Install with optional extras (notebooks, retrieval, alternate solvers)
	cd "$(REPO_ROOT)" && $(UV) sync --all-extras

smoke: ## Quick import check for core modules
	cd "$(REPO_ROOT)" && $(UV) run python -c "import gurobipy; import src.utils; import src.train_eval_utils; print('ok')"

train: ## Run training with TRAIN_CONFIG=... (default: train_config.yaml)
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python main.py --config "$(TRAIN_CONFIG)"

eval: ## Run evaluation with EVAL_CONFIG=... and TRAIN_CONFIG=... for pricing/API context
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(EVAL_CONFIG)"

check-proxy: ## Verify OpenAI-compatible proxy at localhost:8801 (optional local setup)
	@curl -sf http://localhost:8801/health >/dev/null && echo "proxy health OK" || (echo "debug: no proxy at http://localhost:8801/health — use direct API keys or set base_service to your proxy URL in train_config.yaml" >&2; exit 1)

train-with-proxy: check-proxy ## Run training only after proxy health check passes
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python main.py --config "$(TRAIN_CONFIG)"

ensure-e1-library:
	@test -f "$(REPO_ROOT)/$(E1_LIBRARY)" || (echo "missing E1 library: $(E1_LIBRARY)" >&2; exit 1)
	@test -f "$(REPO_ROOT)/$(E1_TAXONOMY)" || (echo "missing E1 taxonomy: $(E1_TAXONOMY)" >&2; exit 1)

ensure-e2-library:
	@test -f "$(REPO_ROOT)/$(E2_LIBRARY)" || (echo "missing E2 library: $(E2_LIBRARY). Run 'make exp-e2' training step first." >&2; exit 1)
	@test -f "$(REPO_ROOT)/$(E2_TAXONOMY)" || (echo "missing E2 taxonomy: $(E2_TAXONOMY). Run 'make exp-e2' training step first." >&2; exit 1)

exp-e1: ensure-e1-library ## E1: evaluate GPT-5.4 on all eval datasets with the preserved original GPT-4o+Gurobi library
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(E1_EVAL_CONFIG)"

exp-e2: ## E2: retrain library with GPT-5.4+Gurobi on original train set, then evaluate on all eval datasets
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python main.py --config "$(E2_TRAIN_CONFIG)"
	@test -f "$(REPO_ROOT)/$(E2_LIBRARY)" || (echo "training finished but expected E2 library was not produced: $(E2_LIBRARY)" >&2; exit 1)
	@test -f "$(REPO_ROOT)/$(E2_TAXONOMY)" || (echo "training finished but expected E2 taxonomy was not produced: $(E2_TAXONOMY)" >&2; exit 1)
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(E2_EVAL_CONFIG)"
