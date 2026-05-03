# Progress log

## [2026-05-02]

- Action: Wired AlphaOPT LLM client to Azure OpenAI via `service: azure`, reading `[azure]` from `~/.config/LLMProxy-credentials.toml` (deployment name used for API calls; optional `AZURE_OPENAI_*` env overrides).
- Result: success
- Notes: `train_config.yaml` / `eval_config.yaml` default to Azure + `gpt-5.4`.

## [2026-05-02]

- Action: Switched `eval_config_e1_gpt54_original.yaml` and `train_config_e2_gpt54_gurobi.yaml` from `http://127.0.0.1:4010/v1` to `service: azure` so `make exp-e1` matches direct Azure credentials (avoids connection errors when no local proxy on 4010).
- Result: success
- Notes: Root cause of reported “Connection error” was unreachable localhost proxy, not Azure SDK parsing.

## [2026-05-02]

- Action: Pushed branch `azure-based` and annotated tag `azure-based` to `origin` (commit Azure LLM/credential wiring).
- Result: success
