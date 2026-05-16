.PHONY: help sync sync-extras smoke train eval check-proxy train-with-proxy exp-e1 exp-e2 ensure-e1-library ensure-e2-library laminar-env laminar-up laminar-down laminar-logs laminar-status laminar-check train-laminar eval-laminar exp-e1-laminar exp-e2-laminar

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
LAMINAR_COMPOSE := infra/docker-compose.laminar.yml
LAMINAR_ENV := infra/.env

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' "$(REPO_ROOT)/Makefile" | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

sync: ## Install locked dependencies (uv sync)
	cd "$(REPO_ROOT)" && $(UV) sync

sync-extras: ## Install with optional extras (notebooks, retrieval, alternate solvers)
	cd "$(REPO_ROOT)" && $(UV) sync --all-extras

smoke: ## Quick import check for core modules
	cd "$(REPO_ROOT)" && $(UV) run python -c "import gurobipy; import src.utils; import src.train_eval_utils; print('ok')"

train: laminar-env ## Run training with TRAIN_CONFIG=... (Laminar enabled by infra/.env)
	cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python main.py --config "$(TRAIN_CONFIG)"

eval: laminar-env ## Run evaluation with EVAL_CONFIG=... and TRAIN_CONFIG=... (Laminar enabled by infra/.env)
	cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(EVAL_CONFIG)"

check-proxy: ## Verify OpenAI-compatible proxy at localhost:8801 (optional local setup)
	@curl -sf http://localhost:8801/health >/dev/null && echo "proxy health OK" || (echo "debug: no proxy at http://localhost:8801/health — use direct API keys or set base_service to your proxy URL in configs/train/default.yaml" >&2; exit 1)

train-with-proxy: check-proxy ## Run training only after proxy health check passes
	cd "$(REPO_ROOT)" && ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python main.py --config "$(TRAIN_CONFIG)"

laminar-env: ## Create infra/.env for the self-hosted Laminar stack
	cd "$(REPO_ROOT)" && $(UV) run python infra/create_laminar_env.py

laminar-up: laminar-env ## Start self-hosted Laminar
	cd "$(REPO_ROOT)" && docker compose --env-file "$(LAMINAR_ENV)" -f "$(LAMINAR_COMPOSE)" up -d --wait

laminar-down: ## Stop self-hosted Laminar
	cd "$(REPO_ROOT)" && docker compose --env-file "$(LAMINAR_ENV)" -f "$(LAMINAR_COMPOSE)" down

laminar-logs: ## Tail Laminar logs
	cd "$(REPO_ROOT)" && docker compose --env-file "$(LAMINAR_ENV)" -f "$(LAMINAR_COMPOSE)" logs -f --tail=100

laminar-status: ## Show Laminar containers
	cd "$(REPO_ROOT)" && docker compose --env-file "$(LAMINAR_ENV)" -f "$(LAMINAR_COMPOSE)" ps

laminar-check: laminar-env ## Validate compose and local Laminar endpoints
	cd "$(REPO_ROOT)" && docker compose --env-file "$(LAMINAR_ENV)" -f "$(LAMINAR_COMPOSE)" config >/dev/null
	@cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; curl -sS --max-time 30 "http://localhost:$${FRONTEND_HOST_PORT:-5667}" >/dev/null && echo "Laminar frontend reachable: http://localhost:$${FRONTEND_HOST_PORT:-5667}" || (echo "Laminar frontend is not reachable. Run 'make laminar-up' first." >&2; exit 1)
	@cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; curl -sS --max-time 10 "http://localhost:$${APP_SERVER_HOST_PORT:-18000}" >/dev/null && echo "Laminar app-server reachable: http://localhost:$${APP_SERVER_HOST_PORT:-18000}" || (echo "Laminar app-server is not reachable. Run 'make laminar-up' first." >&2; exit 1)

train-laminar: laminar-env ## Run training with Laminar tracing enabled
	cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_LAMINAR_ENABLED=1 ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python main.py --config "$(TRAIN_CONFIG)"

eval-laminar: laminar-env ## Run evaluation with Laminar tracing enabled
	cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_LAMINAR_ENABLED=1 ALPHAOPT_TRAIN_CONFIG="$(TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(EVAL_CONFIG)"

exp-e1-laminar: laminar-env ensure-e1-library ## E1 with Laminar tracing enabled
	cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_LAMINAR_ENABLED=1 ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(E1_EVAL_CONFIG)"

exp-e2-laminar: laminar-env ## E2 with Laminar tracing enabled
	@artifact_dir=$$(find "$(REPO_ROOT)/data/experience_library" -maxdepth 1 -mindepth 1 -type d -name 'gpt55_gurobi_train_data_all_452*' -print0 | xargs -0 -r ls -dt 2>/dev/null | while IFS= read -r d; do test -f "$$d/library_refine_iter1.json" && test -f "$$d/latest_taxonomy_refine_iter1.json" && { printf '%s\n' "$$d"; break; }; done); \
	if test -n "$$artifact_dir"; then \
		echo "E2 library already exists: $$artifact_dir; skipping training."; \
	else \
		cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_LAMINAR_ENABLED=1 ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python main.py --config "$(E2_TRAIN_CONFIG)"; \
	fi
	@artifact_dir=$$(find "$(REPO_ROOT)/data/experience_library" -maxdepth 1 -mindepth 1 -type d -name 'gpt55_gurobi_train_data_all_452*' -print0 | xargs -0 -r ls -dt 2>/dev/null | while IFS= read -r d; do test -f "$$d/library_refine_iter1.json" && test -f "$$d/latest_taxonomy_refine_iter1.json" && { printf '%s\n' "$$d"; break; }; done); \
	test -n "$$artifact_dir" || (echo "training finished but no E2 artifact directory was found." >&2; exit 1); \
	test -f "$$artifact_dir/library_refine_iter1.json" || (echo "training finished but expected E2 library was not produced: $$artifact_dir/library_refine_iter1.json" >&2; exit 1); \
	test -f "$$artifact_dir/latest_taxonomy_refine_iter1.json" || (echo "training finished but expected E2 taxonomy was not produced: $$artifact_dir/latest_taxonomy_refine_iter1.json" >&2; exit 1); \
	tmp_cfg=$$(mktemp); \
	sed -e "s#^library_path: .*#library_path: ./data/experience_library/$$(basename "$$artifact_dir")/library_refine_iter1.json#" \
	    -e "s#^taxonomy_path: .*#taxonomy_path: ./data/experience_library/$$(basename "$$artifact_dir")/latest_taxonomy_refine_iter1.json#" \
	    "$(REPO_ROOT)/$(E2_EVAL_CONFIG)" > "$$tmp_cfg"; \
	cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_LAMINAR_ENABLED=1 ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$$tmp_cfg"; \
	rm -f "$$tmp_cfg"

ensure-e1-library:
	@test -f "$(REPO_ROOT)/$(E1_LIBRARY)" || (echo "missing E1 library: $(E1_LIBRARY)" >&2; exit 1)
	@test -f "$(REPO_ROOT)/$(E1_TAXONOMY)" || (echo "missing E1 taxonomy: $(E1_TAXONOMY)" >&2; exit 1)

ensure-e2-library:
	@artifact_dir=$$(find "$(REPO_ROOT)/data/experience_library" -maxdepth 1 -mindepth 1 -type d -name 'gpt55_gurobi_train_data_all_452*' -print0 | xargs -0 -r ls -dt 2>/dev/null | while IFS= read -r d; do test -f "$$d/library_refine_iter1.json" && test -f "$$d/latest_taxonomy_refine_iter1.json" && { printf '%s\n' "$$d"; break; }; done); \
	test -n "$$artifact_dir" || (echo "missing E2 artifact directory. Run 'make exp-e2' training step first." >&2; exit 1); \
	test -f "$$artifact_dir/library_refine_iter1.json" || (echo "missing E2 library: $$artifact_dir/library_refine_iter1.json. Run 'make exp-e2' training step first." >&2; exit 1); \
	test -f "$$artifact_dir/latest_taxonomy_refine_iter1.json" || (echo "missing E2 taxonomy: $$artifact_dir/latest_taxonomy_refine_iter1.json. Run 'make exp-e2' training step first." >&2; exit 1)

exp-e1: laminar-env ensure-e1-library ## E1: evaluate GPT-5.5 on all eval datasets with the preserved default library
	cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$(E1_EVAL_CONFIG)"

exp-e2: laminar-env ## E2: retrain library with GPT-5.5+Gurobi on original train set, then evaluate on all eval datasets
	@artifact_dir=$$(find "$(REPO_ROOT)/data/experience_library" -maxdepth 1 -mindepth 1 -type d -name 'gpt55_gurobi_train_data_all_452*' -print0 | xargs -0 -r ls -dt 2>/dev/null | while IFS= read -r d; do test -f "$$d/library_refine_iter1.json" && test -f "$$d/latest_taxonomy_refine_iter1.json" && { printf '%s\n' "$$d"; break; }; done); \
	if test -n "$$artifact_dir"; then \
		echo "E2 library already exists: $$artifact_dir; skipping training."; \
	else \
		cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python main.py --config "$(E2_TRAIN_CONFIG)"; \
	fi
	@artifact_dir=$$(find "$(REPO_ROOT)/data/experience_library" -maxdepth 1 -mindepth 1 -type d -name 'gpt55_gurobi_train_data_all_452*' -print0 | xargs -0 -r ls -dt 2>/dev/null | while IFS= read -r d; do test -f "$$d/library_refine_iter1.json" && test -f "$$d/latest_taxonomy_refine_iter1.json" && { printf '%s\n' "$$d"; break; }; done); \
	test -n "$$artifact_dir" || (echo "training finished but no E2 artifact directory was found." >&2; exit 1); \
	test -f "$$artifact_dir/library_refine_iter1.json" || (echo "training finished but expected E2 library was not produced: $$artifact_dir/library_refine_iter1.json" >&2; exit 1); \
	test -f "$$artifact_dir/latest_taxonomy_refine_iter1.json" || (echo "training finished but expected E2 taxonomy was not produced: $$artifact_dir/latest_taxonomy_refine_iter1.json" >&2; exit 1); \
	tmp_cfg=$$(mktemp); \
	sed -e "s#^library_path: .*#library_path: ./data/experience_library/$$(basename "$$artifact_dir")/library_refine_iter1.json#" \
	    -e "s#^taxonomy_path: .*#taxonomy_path: ./data/experience_library/$$(basename "$$artifact_dir")/latest_taxonomy_refine_iter1.json#" \
	    "$(REPO_ROOT)/$(E2_EVAL_CONFIG)" > "$$tmp_cfg"; \
	cd "$(REPO_ROOT)" && set -a; . "$(LAMINAR_ENV)"; set +a; ALPHAOPT_TRAIN_CONFIG="$(E2_TRAIN_CONFIG)" $(UV) run python evaluation.py --config "$$tmp_cfg"; \
	rm -f "$$tmp_cfg"
