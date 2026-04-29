.PHONY: help sync sync-extras train eval check-proxy train-with-proxy smoke regen-smoke regen

UV ?= uv
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(REPO_ROOT:/=)

# Defaults for `make regen` / `make regen-smoke`. Override per-invocation, e.g.
#   make regen REGEN_INPUT=path/in.json REGEN_OUTPUT=path/out.json
REGEN_INPUT  ?= data/optimization_tasks/train/train_data_all_452.json
REGEN_OUTPUT ?= data/optimization_tasks/train/train_data_all_452_pulp.json

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' "$(REPO_ROOT)/Makefile" | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

sync: ## Install locked dependencies (uv sync)
	cd "$(REPO_ROOT)" && $(UV) sync

sync-extras: ## Install with optional extras (notebooks, retrieval, legacy-solvers)
	cd "$(REPO_ROOT)" && $(UV) sync --all-extras

smoke: ## Quick import check for core modules
	cd "$(REPO_ROOT)" && $(UV) run python -c "import src.utils; import src.credentials; import pulp; print('ok')"

train: ## Run training (reads train_config.yaml in cwd)
	cd "$(REPO_ROOT)" && $(UV) run python main.py

eval: ## Run evaluation (reads eval_config.yaml in cwd)
	cd "$(REPO_ROOT)" && $(UV) run python evaluation.py

regen-smoke: ## Regenerate correct_program for the first 5 training tasks (PuLP+CBC) — sanity check
	cd "$(REPO_ROOT)" && $(UV) run python -m src.tools.regen_correct_programs \
		--input  "$(REGEN_INPUT)" \
		--output "$(REGEN_OUTPUT)" \
		--limit  5

regen: ## Regenerate correct_program for ALL training tasks (PuLP+CBC) — spends LLM tokens
	cd "$(REPO_ROOT)" && $(UV) run python -m src.tools.regen_correct_programs \
		--input  "$(REGEN_INPUT)" \
		--output "$(REGEN_OUTPUT)"

check-proxy: ## Verify OpenAI-compatible proxy at localhost:8801 (optional local setup)
	@curl -sf http://localhost:8801/health >/dev/null && echo "proxy health OK" || (echo "debug: no proxy at http://localhost:8801/health — use direct API keys or set base_service to your proxy URL in train_config.yaml" >&2; exit 1)

train-with-proxy: check-proxy ## Run training only after proxy health check passes
	cd "$(REPO_ROOT)" && $(UV) run python main.py
