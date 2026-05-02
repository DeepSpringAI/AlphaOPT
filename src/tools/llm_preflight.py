"""
Quick LLM sanity check before train / regen / eval.

Reads ``advanced_model`` + ``advanced_service`` (and ``api_keys``) from the given YAML
(default ``train_config.yaml``), matches what the rest of the pipeline uses by assigning
``src.utils.config`` from that file, optionally GETs ``/health`` on local proxy URLs, then
runs one minimal chat completion and prints latency.

Usage::
    cd /path/to/AlphaOPT && uv run python -m src.tools.llm_preflight
    make preflight-llm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _health_url_from_openai_base(base_service: str) -> str | None:
    """``http://127.0.0.1:8801/v1`` -> ``http://127.0.0.1:8801/health``."""
    b = (base_service or "").strip().rstrip("/")
    if not (b.startswith("http://") or b.startswith("https://")):
        return None
    if b.endswith("/v1"):
        root = b[:-3].rstrip("/")
    else:
        root = b
    return root + "/health"


def _probe_health(health_url: str, timeout: float) -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - t0
        try:
            data = json.loads(body)
            extra = data.get("status", body[:120])
        except json.JSONDecodeError:
            extra = body[:120]
        return True, elapsed, f"OK ({extra})"
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        return False, elapsed, f"HTTP {e.code}"
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return False, elapsed, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="train_config.yaml",
        help="YAML with advanced_model / advanced_service / api_keys (default: train_config.yaml).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="OpenAI SDK timeout for the chat request (seconds).",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=8.0,
        help="HTTP timeout for GET /health (seconds).",
    )
    parser.add_argument(
        "--warn-latency",
        type=float,
        default=45.0,
        help="Print a warning if chat round-trip exceeds this many seconds (still exits 0).",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Do not GET /health before chat (only use when proxy has no health route).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"preflight: config not found: {cfg_path.resolve()}", file=sys.stderr)
        return 1

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(cfg_path.resolve()))
    OmegaConf.resolve(cfg)

    # Align credential lookup with the same YAML train/regen use (``src.utils.config``).
    import src.utils as su

    su.config = cfg

    from src.utils import (
        _apply_gpt5_openai_chat_params,
        _build_client,
        _upstream_llm_error_hint,
    )

    model = str(cfg.advanced_model)
    service = str(cfg.advanced_service) if cfg.advanced_service is not None else "null"

    print(f"preflight: config={cfg_path.name} model={model!r} service={service!r}")

    if Path.cwd().resolve() != _REPO_ROOT.resolve():
        print(
            f"preflight: warning: cwd {Path.cwd()} is not repo root {_REPO_ROOT} "
            "(Makefile ``cd`` avoids this).",
            file=sys.stderr,
        )

    if not args.skip_health:
        hu = _health_url_from_openai_base(service)
        if hu:
            ok, sec, msg = _probe_health(hu, args.health_timeout)
            print(f"preflight: GET {hu} -> {sec * 1000:.0f} ms — {msg}")
            if not ok:
                print(
                    "preflight: proxy health check failed (fix llm-api-proxy / MLX before long runs).",
                    file=sys.stderr,
                )
                return 1
        else:
            print("preflight: skip GET /health (service is not an http(s) OpenAI base URL).")

    try:
        vendor, client = _build_client(model, service, timeout_sec=args.timeout)
    except Exception as e:
        print(f"preflight: _build_client failed: {e}", file=sys.stderr)
        print(_upstream_llm_error_hint(e), file=sys.stderr)
        return 1

    if vendor == "gemini":
        from google.genai import types as genai_types

        t0 = time.perf_counter()
        client.models.generate_content(
            model=model,
            contents="Reply with exactly: pong",
            config=genai_types.GenerateContentConfig(max_output_tokens=32, temperature=0.2),
        )
        elapsed = time.perf_counter() - t0
        print(f"preflight: gemini generate_content OK in {elapsed * 1000:.0f} ms")
        if elapsed > args.warn_latency:
            print(
                f"preflight: warning: latency {elapsed:.1f}s exceeds --warn-latency {args.warn_latency}s",
                file=sys.stderr,
            )
        return 0

    if vendor not in ("openai", "openrouter"):
        print(f"preflight: unsupported vendor {vendor!r} for chat probe", file=sys.stderr)
        return 1

    msgs = [{"role": "user", "content": "Reply with exactly: pong"}]
    create_params: dict = {
        "model": model,
        "messages": msgs,
        "max_tokens": 64,
        "temperature": 1.0,
    }
    if vendor == "openai":
        create_params["frequency_penalty"] = 0.5
    create_params = _apply_gpt5_openai_chat_params(model, vendor, create_params)

    t0 = time.perf_counter()
    try:
        completion = client.chat.completions.create(**create_params)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"preflight: chat failed after {elapsed * 1000:.0f} ms: {e}", file=sys.stderr)
        print(_upstream_llm_error_hint(e), file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    text = (completion.choices[0].message.content or "").strip()
    preview = text.replace("\n", " ")[:160]
    print(f"preflight: chat.completions OK in {elapsed * 1000:.0f} ms — preview: {preview!r}")

    if elapsed > args.warn_latency:
        print(
            f"preflight: warning: latency {elapsed:.1f}s exceeds --warn-latency {args.warn_latency}s",
            file=sys.stderr,
        )

    host = urlparse(service).hostname if service.startswith("http") else ""
    if host in ("127.0.0.1", "localhost") and elapsed > args.warn_latency:
        print(
            "preflight: hint: high latency often means MLX/upstream slowness or overload (not local CPU).",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
