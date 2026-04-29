"""
Load optional API keys from ~/.config/AlphaOPT-credentials.toml (TOML).
Used together with YAML config and environment variables in src.utils._build_client.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

CREDENTIALS_PATH = Path.home() / ".config" / "AlphaOPT-credentials.toml"

_toml_cache: dict[str, str] | None = None


def _flatten_api_keys_table(raw: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    keys = raw.get("api_keys")
    if isinstance(keys, dict):
        for k, v in keys.items():
            if isinstance(v, str) and v.strip():
                out[str(k)] = v.strip()
    return out


def load_credentials_toml() -> dict[str, str]:
    """Return OPEN_ROUTER_KEY / GEMINI_API_KEY / OPENAI_API_KEY from TOML if present."""
    global _toml_cache
    if _toml_cache is not None:
        return _toml_cache
    _toml_cache = {}
    path = CREDENTIALS_PATH.expanduser().resolve()
    if not path.is_file():
        return _toml_cache
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return _toml_cache
    if not isinstance(data, dict):
        return _toml_cache
    _toml_cache = _flatten_api_keys_table(data)
    return _toml_cache


def invalidate_credentials_cache() -> None:
    global _toml_cache
    _toml_cache = None
