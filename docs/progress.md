# Progress

Append-only log for meaningful changes.

## [2026-04-30 13:00]

- Action: Added **`make preflight-llm`** / **`make check-llm`** (`src/tools/llm_preflight.py`): GET proxy **`/health`**, timed **`chat.completions`** “pong” probe using **`train_config.yaml`**, shared **`src.utils.config`** for keys; optional flags (`--timeout`, `--warn-latency`, `--skip-health`). Extended **`_build_client(..., timeout_sec=)`** for OpenAI clients.
- Result: success (local run: health ~13 ms, chat ~4.7 s)

## [2026-04-30 12:00]

- Action: Documented regen failures from **502 Bad Gateway (nginx → LiteLLM/MLX)** and occasional **prompt-injection** rejections as **upstream/infra** (not local AlphaOPT). Renamed/expanded `src/utils.py` hint helper to `_upstream_llm_error_hint` (502 + injection + existing key hints).
- Result: success

## [2026-04-30 11:20]

- Action: Set `train_config.yaml` to **`gpt-5.4`** (per QA row in `llm-api-proxy/docs/MODEL_PROXY_MATRIX_REPORT.md`). Fixed **`make regen` log capture**: `PYTHONUNBUFFERED=1` in Makefile + `flush=True` on regen prints (stdout was block-buffered when piped to `tee`). Full run logs to **`docs/regen-gpt-5.4-qa.log`** (includes aborted first attempt + restart).
- Result: success (regen running / monitor with `tail -f docs/regen-gpt-5.4-qa.log`)

## [2026-04-30]

- Action: Diagnosed `make regen` failures as LiteLLM HTTP 500 — upstream API key **expired** (not regen script bugs). Added `_upstream_auth_error_hint()` in `src/utils.py` and clarified Makefile `regen` help.
- Result: success

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
