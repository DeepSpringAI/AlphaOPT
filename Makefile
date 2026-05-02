.PHONY: help sync sync-extras smoke train eval check-proxy train-with-proxy

UV ?= uv
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(REPO_ROOT:/=)

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' "$(REPO_ROOT)/Makefile" | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

sync: ## Install locked dependencies (uv sync)
	cd "$(REPO_ROOT)" && $(UV) sync

sync-extras: ## Install with optional extras (notebooks, retrieval, alternate solvers)
	cd "$(REPO_ROOT)" && $(UV) sync --all-extras

smoke: ## Quick import check for core modules
	cd "$(REPO_ROOT)" && $(UV) run python -c "import gurobipy; import src.utils; import src.train_eval_utils; print('ok')"

train: ## Run training (reads train_config.yaml in cwd)
	cd "$(REPO_ROOT)" && $(UV) run python main.py

eval: ## Run evaluation (reads eval_config.yaml in cwd)
	cd "$(REPO_ROOT)" && $(UV) run python evaluation.py

check-proxy: ## Verify OpenAI-compatible proxy at localhost:8801 (optional local setup)
	@curl -sf http://localhost:8801/health >/dev/null && echo "proxy health OK" || (echo "debug: no proxy at http://localhost:8801/health — use direct API keys or set base_service to your proxy URL in train_config.yaml" >&2; exit 1)

train-with-proxy: check-proxy ## Run training only after proxy health check passes
	cd "$(REPO_ROOT)" && $(UV) run python main.py
