import os
import json
import shutil
import time
from pathlib import Path

from omegaconf import OmegaConf


TRAIN_CONFIG_ENV = "ALPHAOPT_TRAIN_CONFIG"
DEFAULT_TRAIN_CONFIG = "configs/train/default.yaml"
DEFAULT_EXPERIENCE_LIBRARY_ROOT = "data/experience_library"
DEFAULT_SHARED_SUBDIR = "shared"


def get_train_config_path() -> str:
    return os.getenv(TRAIN_CONFIG_ENV, DEFAULT_TRAIN_CONFIG)


def load_train_config():
    return OmegaConf.load(get_train_config_path())


def get_training_library_subdir_name(config_path: str, config=None) -> str:
    configured = ""
    if config is not None:
        configured = str(getattr(config, "library_subdir", "") or "").strip()
    if configured:
        return configured
    return Path(config_path).stem


def _next_available_subdir(root: Path, base_name: str) -> str:
    candidate = base_name
    index = 1
    while True:
        candidate_dir = root / candidate
        if not candidate_dir.exists() or not any(candidate_dir.iterdir()):
            return candidate
        candidate = f"{base_name}_{index}"
        index += 1


def get_experience_library_root(config) -> Path:
    configured_root = str(getattr(config, "experience_library_root", "") or "").strip()
    lib_dir = configured_root or str(config.file_paths.lib_dir)
    root = Path(lib_dir).resolve()
    return root


def get_shared_artifact_dir(config) -> Path:
    shared_subdir = str(getattr(config, "shared_artifact_subdir", DEFAULT_SHARED_SUBDIR) or DEFAULT_SHARED_SUBDIR).strip()
    return get_experience_library_root(config) / shared_subdir


def prepare_training_artifact_paths(config, *, config_path: str):
    """
    Scope training artifacts to a dedicated experience_library subdirectory named
    after the train config (or an explicit library_subdir override).
    """
    library_root_dir = get_experience_library_root(config)
    requested_subdir = get_training_library_subdir_name(config_path, config=config)
    library_subdir = requested_subdir
    artifact_dir = library_root_dir / library_subdir
    start_iter = int(getattr(config, "start_iter", 0) or 0)

    if start_iter == 0:
        library_subdir = _next_available_subdir(library_root_dir, requested_subdir)
        artifact_dir = library_root_dir / library_subdir
    else:
        if not artifact_dir.exists():
            raise ValueError(
                f"Cannot resume training because the artifact subdirectory does not exist: {artifact_dir}"
            )

    artifact_dir.mkdir(parents=True, exist_ok=True)

    config.experience_library_root = str(library_root_dir)
    config.library_subdir = library_subdir
    config.file_paths.lib_dir = str(artifact_dir)
    return config, artifact_dir


def copy_shared_artifacts_to_training_subdir(config) -> list[str]:
    """
    Copy human-authored shared artifacts into the dedicated training subdirectory.
    """
    shared_dir = get_shared_artifact_dir(config)
    artifact_dir = Path(str(config.file_paths.lib_dir))
    copied_paths: list[str] = []

    if not shared_dir.exists():
        return copied_paths

    for src in sorted(shared_dir.iterdir()):
        if not src.is_file():
            continue
        dst = artifact_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied_paths.append(str(dst))

    return copied_paths


def get_seed_taxonomy_path(config) -> str:
    artifact_dir = Path(str(config.file_paths.lib_dir))
    local_seed = artifact_dir / "fewshot_taxonomy.json"
    if local_seed.exists():
        return str(local_seed)
    shared_seed = get_shared_artifact_dir(config) / "fewshot_taxonomy.json"
    return str(shared_seed)


def write_training_run_metadata(config, *, config_path: str) -> str:
    """
    Persist a small metadata/config snapshot inside the experience-library subdirectory.
    """
    artifact_dir = Path(str(config.file_paths.lib_dir))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = OmegaConf.to_container(config, resolve=True)
    metadata = {
        "library_subdir": str(config.library_subdir),
        "config_path": config_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": resolved_cfg.get("dataset"),
        "output_folder": resolved_cfg.get("output_folder"),
        "lib_dir": resolved_cfg.get("file_paths", {}).get("lib_dir"),
        "train_output_dir": resolved_cfg.get("file_paths", {}).get("train_output_dir"),
        "shared_artifact_subdir": resolved_cfg.get("shared_artifact_subdir", DEFAULT_SHARED_SUBDIR),
        "models": {
            "base_model": resolved_cfg.get("base_model"),
            "advanced_model": resolved_cfg.get("advanced_model"),
            "base_service": resolved_cfg.get("base_service"),
            "advanced_service": resolved_cfg.get("advanced_service"),
        },
    }

    metadata_path = artifact_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    config_snapshot_path = artifact_dir / "train_config.resolved.json"
    with config_snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2, ensure_ascii=False)

    return str(metadata_path)
