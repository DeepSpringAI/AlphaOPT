# Progress

Append-only log for meaningful changes.

## [2026-04-29 16:00]

- Action: Debugged `make train` — LLM client builds for `gpt-5` + `http://127.0.0.1:8801/v1`; proxy `/health` OK. Runtime task failures are **Gurobi** `License expired 2025-11-14` when executing generated programs, not OpenAI/proxy import errors.
- Result: success (root cause identified)
- Notes: Renew or replace Gurobi license (e.g. `grbgetkey` / `GRB_LICENSE_FILE`). Optional: align llm-api-proxy `active_backend` / model with `gpt-5` if chat routing is wrong.

## [2026-04-29 15:00]

- Action: Renamed model id from `gpt-5.4` to `gpt-5` across train/eval YAML, pricing key, and output folder names (`train_data_all_452_gpt5`, `*_eval_gpt5`).
- Result: success

## [2026-04-29 14:30]

- Action: Standardized all training and evaluation YAML on `gpt-5.4` with `http://127.0.0.1:8801/v1` (llm-api-proxy). Added `_is_gpt5_family_model_id` / `_apply_gpt5_openai_chat_params` in `src/utils.py` so chat calls use temperature 1.0 and no `frequency_penalty` for GPT-5 family, matching proxy/Azure behavior.
- Result: success
- Notes: Proxy must route request model to the real deployment (e.g. Azure `deployment_name` in LLMProxy when `active_backend` is `azure`). `train_config` `output_folder` was `train_data_all_452_gpt54` (see later entry for `gpt-5` rename).

## [2026-04-29 12:00]

- Action: Adopted workspace Python conventions — `pyproject.toml` + `uv.lock`, `Makefile` targets (`sync`, `train`, `eval`, `check-proxy`, `smoke`), and optional API keys via `~/.config/AlphaOPT-credentials.toml` (see `config/AlphaOPT-credentials.example.toml`). Removed root `requirements.txt` in favor of UV.
- Result: success
- Notes: `make smoke` passes; key precedence is env → YAML `api_keys` → TOML file.

## [2026-04-29]

- Action: Debugged `make train`: fixed `FileNotFoundError` when saving checkpoints if the experience library is empty (output dirs were only created inside `library.save`, which is skipped for an empty library). `save_checkpoint` now creates `train_output_dir`, `lib_dir`, and the metrics log parent. Clarified `OPEN_ROUTER_KEY` error message to mention TOML path.
- Result: success
- Notes: Real training still requires `OPEN_ROUTER_KEY` (or proxy URL in `base_service`/`advanced_service` per README) when using OpenRouter.
