# Configs

- `configs/train/default.yaml`
  - Default training config.
- `configs/train/gpt55_gurobi_train_data_all_452.yaml`
  - Named training config for the GPT-5.5 + Gurobi setup.
- `configs/eval/default.yaml`
  - Default evaluation config.
- `configs/eval/gpt55_original_library.yaml`
  - Evaluation config for the preserved default library.
- `configs/eval/gpt55_retrained_library.yaml`
  - Evaluation config for the retrained library.

Keep new configs here with short descriptive filenames.

Training writes experience-library artifacts into a subdirectory named after the train config:

- `configs/train/default.yaml` writes to `data/experience_library/default/`
- `configs/train/gpt55_gurobi_train_data_all_452.yaml` writes to `data/experience_library/gpt55_gurobi_train_data_all_452/`

The preserved shared assets live in:

- `data/experience_library/shared/`
  - human-authored seed files such as `fewshot_taxonomy.json`

The preserved canonical baseline artifacts live in:

- `data/experience_library/default/`
  - `library.json`
  - `latest_taxonomy_new.json` (paper-aligned default)
  - `latest_taxonomy.json` (legacy)

Each training subdirectory keeps the copied shared seed artifacts, final library, taxonomy snapshots, iterations, and run metadata together.

If a fresh run would collide with an existing non-empty subdirectory, AlphaOPT automatically uses `_<index>` suffixes such as `default_1`.
