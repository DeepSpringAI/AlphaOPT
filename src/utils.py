import os
import re
import time
import json
import hashlib
import secrets
import random
from typing import Any, Callable, Dict
from copy import deepcopy
import threading
from pathlib import Path
from dotenv import load_dotenv

#* Configure
from .config import load_train_config

config = load_train_config()


# ==== Global token usage tracker ====
TOKEN_USAGE: Dict[str, Dict[str, float]] = {
    "openai": {
        "requests": 0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "cost": 0.0,
    },
    "openrouter": {
        "requests": 0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "cost": 0.0,
    },
    "gemini": {
        "requests": 0,
        "prompt_tokens": 0.0,   # from count_tokens
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "cost": 0.0,
    },
}

# Protect TOKEN_USAGE updates under multithreading.
_TOKEN_USAGE_LOCK = threading.Lock()
_TRACE_WRITE_LOCK = threading.Lock()
_TRACE_SESSION_LOCK = threading.Lock()
_TRACE_SESSION_DIR: Path | None = None
_TRACE_SESSION_RUN_ID: str | None = None
_EXPERIMENT_META_LOCK = threading.Lock()
_EXPERIMENT_META_CACHE: dict | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "off", "no")


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(str(raw).strip())
    except Exception:
        return default
    return max(min_value, val)


def _env_float(name: str, default: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(str(raw).strip())
    except Exception:
        return default
    return max(min_value, min(max_value, val))


def _trace_io_mode() -> str:
    """
    one of: none | errors | all
    """
    mode = str(os.getenv("ALPHAOPT_TRACE_IO_MODE", "errors")).strip().lower()
    if mode not in ("none", "errors", "all"):
        return "errors"
    return mode


def _should_capture_io(*, on_error: bool) -> bool:
    mode = _trace_io_mode()
    if mode == "none":
        return False
    if mode == "all":
        return True
    return on_error


def _truncate_text_for_storage(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    suffix = f"\n\n[TRUNCATED_FOR_STORAGE omitted_chars={omitted}]"
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix, True


def _compact_dict(data: dict | None) -> dict:
    """
    Remove None/empty-string values to keep trace lines compact.
    """
    if not data:
        return {}
    out: dict[str, Any] = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        out[k] = v
    return out


def _should_write_span(ok: bool) -> bool:
    """
    Keep all error spans, optionally sample successful spans.
    """
    if not ok:
        return True
    rate = _env_float("ALPHAOPT_TRACE_SPAN_SAMPLE_RATE", 1.0, 0.0, 1.0)
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def _record_usage(
    vendor: str,
    prompt_tokens: float | None = None,
    completion_tokens: float | None = None,
    total_tokens: float | None = None,
    cost: float | None = None,
) -> None:
    """
    Update global TOKEN_USAGE in-place.
    """
    if vendor not in TOKEN_USAGE:
        return
    with _TOKEN_USAGE_LOCK:
        usage = TOKEN_USAGE[vendor]
        usage["requests"] += 1
        if prompt_tokens is not None:
            usage["prompt_tokens"] += float(prompt_tokens)
        if completion_tokens is not None:
            usage["completion_tokens"] += float(completion_tokens)
        if total_tokens is not None:
            usage["total_tokens"] += float(total_tokens)
        if cost is not None:
            usage["cost"] += float(cost)


def _estimate_cost(
    vendor: str,
    model: str,
    prompt_tokens: float | None,
    completion_tokens: float | None,
) -> float:
    """
    Estimate USD cost for a single request based on token counts and config.pricing.
    """
    pricing = getattr(config, "pricing", None)
    if pricing is None:
        return 0.0

    prompt_tokens = float(prompt_tokens or 0.0)
    completion_tokens = float(completion_tokens or 0.0)

    # Get vendor config (gemini / openai / openrouter)
    vendor_cfg = getattr(pricing, vendor, None)
    if vendor_cfg is None:
        return 0.0

    # Try exact model key first
    model_cfg = getattr(vendor_cfg, model, None)
    # For OpenRouter we may not know concrete model → use "default"
    if model_cfg is None and vendor == "openrouter":
        model_cfg = getattr(vendor_cfg, "default", None)

    if model_cfg is None:
        return 0.0

    p_per_m = float(getattr(model_cfg, "prompt_per_million", 0.0))
    c_per_m = float(getattr(model_cfg, "completion_per_million", 0.0))

    prompt_cost = (prompt_tokens / 1_000_000.0) * p_per_m
    completion_cost = (completion_tokens / 1_000_000.0) * c_per_m
    return prompt_cost + completion_cost


def get_token_usage() -> Dict[str, Dict[str, float]]:
    """
    Return a deep copy of current global token usage snapshot.
    """
    with _TOKEN_USAGE_LOCK:
        return deepcopy(TOKEN_USAGE)


def reset_token_usage() -> None:
    """
    Reset global token usage counters to zero.
    """
    global TOKEN_USAGE
    with _TOKEN_USAGE_LOCK:
        for vendor, stats in TOKEN_USAGE.items():
            for k in stats.keys():
                stats[k] = 0.0

def _build_client(model: str, service: str):
    """
    Return (vendor, client) according to the model name
    """
    load_dotenv()
    
    # Get API keys from config or fallback to environment variables
    openrouter_key = config.api_keys.OPEN_ROUTER_KEY or os.getenv("OPEN_ROUTER_KEY")
    gemini_key = config.api_keys.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    openai_key = config.api_keys.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    
    if service and service.lower() != "null":
        from openai import OpenAI

        service_lower = service.lower()

        if service_lower == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            if not openrouter_key:
                raise RuntimeError("OPEN_ROUTER_KEY is not set in config or environment")
            client = OpenAI(api_key=openrouter_key, base_url=base_url)
            return "openrouter", client

        # Allow a custom OpenAI-compatible endpoint such as a local vLLM / LM Studio / proxy server.
        if service_lower.startswith("http://") or service_lower.startswith("https://"):
            # Some local OpenAI-compatible servers do not enforce auth; the OpenAI SDK still
            # requires an API key argument, so fall back to a harmless placeholder.
            api_key = openai_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
            client = OpenAI(api_key=api_key, base_url=service)
            return "openai", client

        raise RuntimeError(
            f"Unsupported service value: {service}. Use 'openrouter', 'null', or a full http(s) OpenAI-compatible base URL."
        )

    # Gemini family
    if "gemini" in model.lower():
        # api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not set in config or environment")
        from google import genai
        client = genai.Client(api_key=gemini_key)
        return "gemini", client

    if "gpt" in model.lower():
        # OpenAI-compatible (includes vLLM)
        from openai import OpenAI
        # api_key = os.getenv("OPENAI_API_KEY")
        # if not api_key:
        #     raise RuntimeError("OPENAI_API_KEY is not set")
        # client = OpenAI(api_key=api_key)
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is not set in config or environment")
        client = OpenAI(api_key=openai_key)
        return "openai", client
    
    # Default case - should not reach here
    raise RuntimeError(f"Unsupported model: {model}")


def _prompt_to_text(prompt: Any) -> str:
    """
    Convert prompt payloads (string or OpenAI-style message list) into plain text.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        chunks = []
        for msg in prompt:
            if isinstance(msg, dict):
                role = str(msg.get("role", ""))
                content = msg.get("content", "")
                chunks.append(f"[{role}] {content}")
            else:
                chunks.append(str(msg))
        return "\n\n".join(chunks)
    return str(prompt or "")


def _extract_suspect_terms(text: str, max_terms: int = 20) -> list[str]:
    """
    Heuristic term extractor to help identify potentially moderated wording.
    """
    low = (text or "").lower()
    watch_terms = [
        "morphine", "opioid", "painkiller", "sleeping pill", "pill", "drug",
        "medication", "poison", "suicide", "self-harm", "harm", "kill",
        "death", "overdose", "blood", "weapon", "attack", "violence",
    ]
    hits = []
    for term in watch_terms:
        if term in low:
            hits.append(term)
        if len(hits) >= max_terms:
            break
    return hits


def _parse_content_filter_flags(err_text: str) -> dict:
    """
    Best-effort parser for Azure content_filter_result fields in error messages.
    """
    flags: dict[str, dict[str, Any]] = {}
    for cat in ("hate", "self_harm", "sexual", "violence", "jailbreak"):
        m = re.search(rf"'{cat}'\s*:\s*\{{([^}}]*)\}}", err_text)
        if not m:
            continue
        block = m.group(1)
        filtered_m = re.search(r"'filtered'\s*:\s*(True|False)", block)
        severity_m = re.search(r"'severity'\s*:\s*'([^']+)'", block)
        detected_m = re.search(r"'detected'\s*:\s*(True|False)", block)
        flags[cat] = {
            "filtered": (filtered_m.group(1) == "True") if filtered_m else None,
            "severity": severity_m.group(1) if severity_m else None,
            "detected": (detected_m.group(1) == "True") if detected_m else None,
        }
    return flags


def _classify_llm_error(err: Exception) -> dict:
    """
    Classify provider errors to improve observability and error messages.
    """
    err_text = repr(err)
    low = err_text.lower()
    status_code = getattr(err, "status_code", None)
    if status_code is None:
        response = getattr(err, "response", None)
        status_code = getattr(response, "status_code", None)

    is_content_filter = (
        "content_filter" in low
        or "responsibleaipolicyviolation" in low
        or "content management policy" in low
    )
    if is_content_filter:
        kind = "content_filter"
    elif "connection error" in low:
        kind = "connection_error"
    elif "rate limit" in low:
        kind = "rate_limit"
    elif "timeout" in low:
        kind = "timeout"
    elif "badrequesterror" in low or "error code: 400" in low:
        kind = "bad_request"
    else:
        kind = "unknown"

    return {
        "kind": kind,
        "status_code": status_code,
        "is_content_filter": is_content_filter,
        "content_filter_result": _parse_content_filter_flags(err_text) if is_content_filter else {},
        "error_repr": err_text,
    }


def _write_llm_error_trace(
    *,
    trace_output_path: str,
    prompt_text: str,
    payload: dict,
) -> dict:
    """
    Persist a compact local trace artifact for failed LLM calls.
    """
    trace_dir = Path(trace_output_path) / "_llm_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    ts = time.strftime("%Y%m%d-%H%M%S")
    task_id = str(payload.get("task_id", "unknown"))
    stage = str(payload.get("stage", "na"))
    op = str(payload.get("operation", "llm_call"))
    attempt = str(payload.get("attempt", "x"))
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{ts}_{task_id}_{stage}_{op}_a{attempt}")

    json_path = trace_dir / f"{base}.json"
    existing_prompt_id = str(payload.get("prompt_id") or "").strip()
    prompt_path = trace_dir / f"{base}.prompt.txt"
    prompt_id = existing_prompt_id or prompt_path.name

    if not existing_prompt_id:
        max_prompt_chars = _env_int("ALPHAOPT_TRACE_MAX_ERROR_PROMPT_CHARS", 12000, 0)
        prompt_for_storage, prompt_truncated = _truncate_text_for_storage(prompt_text, max_prompt_chars)
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_for_storage)
    else:
        prompt_truncated = False

    payload = dict(payload)
    payload["prompt_sha256"] = prompt_hash
    payload["prompt_chars"] = len(prompt_text)
    payload["prompt_truncated"] = prompt_truncated
    payload["suspect_terms"] = _extract_suspect_terms(prompt_text)
    payload["prompt_id"] = prompt_id
    payload["timestamp"] = ts

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"trace_json_id": json_path.name, "trace_prompt_id": prompt_id}


def _to_otlp_status_code(ok: bool) -> str:
    return "STATUS_CODE_OK" if ok else "STATUS_CODE_ERROR"


def _new_trace_id_hex() -> str:
    # OTLP trace_id is 16 bytes => 32 hex chars.
    return secrets.token_hex(16)


def _new_span_id_hex() -> str:
    # OTLP span_id is 8 bytes => 16 hex chars.
    return secrets.token_hex(8)


def _to_primitive(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _callable_name(fn: Any) -> str:
    if fn is None:
        return "unknown_parser"
    name = getattr(fn, "__name__", None)
    if name:
        return str(name)
    wrapped = getattr(fn, "func", None)
    wrapped_name = getattr(wrapped, "__name__", None)
    if wrapped_name:
        return str(wrapped_name)
    return fn.__class__.__name__


def _get_nested(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for p in path.split("."):
        if cur is None:
            return default
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf
        except Exception:
            DictConfig = ListConfig = tuple()  # type: ignore[assignment]
            OmegaConf = None  # type: ignore[assignment]

        if OmegaConf is not None and isinstance(cur, (DictConfig, ListConfig)):
            next_value = OmegaConf.select(cur, p, default=default, throw_on_resolution_failure=False)
            if next_value is default:
                return default
            cur = next_value
        elif isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            return default
    return cur


def _collect_config_meta(path: str, prefix: str) -> dict:
    """
    Load a YAML config and emit lightweight experiment metadata fields.
    """
    out: dict[str, Any] = {}
    p = Path(path)
    out[f"experiment.{prefix}.config_name"] = p.name if p.name else None
    out[f"experiment.{prefix}.config_exists"] = p.exists()
    if not p.exists():
        return out

    try:
        raw = p.read_bytes()
        out[f"experiment.{prefix}.config_sha256"] = hashlib.sha256(raw).hexdigest()
    except Exception:
        out[f"experiment.{prefix}.config_sha256"] = None

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(p))
    except Exception as e:
        out[f"experiment.{prefix}.config_load_error"] = str(e)
        return out

    # Common keys across train/eval style YAMLs
    keys = {
        "model": "model",
        "service": "service",
        "base_model": "base_model",
        "advanced_model": "advanced_model",
        "base_service": "base_service",
        "advanced_service": "advanced_service",
        "dataset": "dataset",
        "datasets": "datasets",
        "output_folder": "output_folder",
        "output_folder_template": "output_folder_template",
        "library_path": "library_path",
        "taxonomy_path": "taxonomy_path",
        "n_runs": "n_runs",
        "temperature": "temperature",
        "pass_at_k": "pass_at_k",
        "start_iter": "start_iter",
    }
    for out_key, cfg_key in keys.items():
        out[f"experiment.{prefix}.{out_key}"] = _to_primitive(_get_nested(cfg, cfg_key))

    # Selected nested train params/ablation fields
    nested_keys = {
        "params.num_iterations": "params.num_iterations",
        "params.batch_size": "params.batch_size",
        "params.max_solution_attempts": "params.max_solution_attempts",
        "ablation.taxonomy": "ablation.taxonomy",
        "ablation.rewrite": "ablation.rewrite",
        "ablation.include_example": "ablation.include_example",
        "ablation.include_program_insight": "ablation.include_program_insight",
        "ablation.max_debug_retry": "ablation.max_debug_retry",
    }
    for out_key, cfg_key in nested_keys.items():
        out[f"experiment.{prefix}.{out_key}"] = _to_primitive(_get_nested(cfg, cfg_key))

    return out


def _get_experiment_metadata() -> dict:
    """
    Build and cache experiment-level metadata from env-configured YAML paths.
    """
    global _EXPERIMENT_META_CACHE
    with _EXPERIMENT_META_LOCK:
        if _EXPERIMENT_META_CACHE is not None:
            return _EXPERIMENT_META_CACHE

        meta: dict[str, Any] = {
            "experiment.release": str(os.getenv("ALPHAOPT_RELEASE", "dev")),
            "experiment.environment": str(os.getenv("ALPHAOPT_ENV", "local")),
        }

        train_cfg = os.getenv("ALPHAOPT_TRAIN_CONFIG")
        eval_cfg = os.getenv("ALPHAOPT_EVAL_CONFIG")

        if train_cfg:
            meta.update(_collect_config_meta(train_cfg, "train"))
        else:
            meta["experiment.train.config_name"] = None
            meta["experiment.train.config_exists"] = False

        if eval_cfg:
            meta.update(_collect_config_meta(eval_cfg, "eval"))
        else:
            meta["experiment.eval.config_name"] = None
            meta["experiment.eval.config_exists"] = False

        _EXPERIMENT_META_CACHE = _compact_dict(meta)
        return meta


def _is_trace_enabled() -> bool:
    v = str(os.getenv("ALPHAOPT_TRACE_ENABLED", "1")).strip().lower()
    return v not in ("0", "false", "off", "no")


def _get_trace_session_dir() -> tuple[Path, str]:
    """
    Return a process-wide trace session directory and run_id.
    """
    global _TRACE_SESSION_DIR, _TRACE_SESSION_RUN_ID
    with _TRACE_SESSION_LOCK:
        if _TRACE_SESSION_DIR is None or _TRACE_SESSION_RUN_ID is None:
            base = Path(os.getenv("ALPHAOPT_TRACE_DIR", "./traces"))
            ts = time.strftime("%Y%m%d-%H%M%S")
            run_id = f"{ts}-pid{os.getpid()}-{secrets.token_hex(4)}"
            session_dir = base / run_id
            session_dir.mkdir(parents=True, exist_ok=True)
            _TRACE_SESSION_DIR = session_dir
            _TRACE_SESSION_RUN_ID = run_id
            print(f"[Tracing] Local OTLP-friendly traces enabled: {session_dir}")
        return _TRACE_SESSION_DIR, _TRACE_SESSION_RUN_ID


def _resolve_trace_output_path(trace_output_path: str | None) -> tuple[str | None, str | None]:
    """
    Resolve the output path used by trace writers.
    If no path is provided, use the global session trace directory.
    """
    if not _is_trace_enabled():
        return None, None
    if trace_output_path:
        return trace_output_path, None
    session_dir, run_id = _get_trace_session_dir()
    return str(session_dir), run_id


def _extract_trace_context_from_log_header(log_header: str | None) -> dict:
    """
    Best-effort context extraction from existing log headers.
    """
    if not log_header:
        return {}
    ctx: dict[str, Any] = {}
    m_iter = re.search(r"\[Iteration\s+([0-9]+)\]", log_header, re.IGNORECASE)
    if m_iter:
        ctx["iteration"] = int(m_iter.group(1))
    m_stage = re.search(r"\[(Formulation|Program|Diagnosis)\]", log_header, re.IGNORECASE)
    if m_stage:
        ctx["stage"] = m_stage.group(1)
    m_task = re.search(r"Task\s+([A-Za-z0-9_-]+)", log_header, re.IGNORECASE)
    if m_task:
        ctx["task_id"] = m_task.group(1)
    return ctx


def _write_llm_io_artifacts(
    *,
    trace_output_path: str,
    prompt_text: str,
    response_text: str | None,
    meta: dict,
) -> dict:
    """
    Persist prompt/response artifacts for each LLM invocation.
    """
    trace_dir = Path(trace_output_path) / "_llm_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    task_id = str(meta.get("task_id", "unknown"))
    stage = str(meta.get("stage", "na"))
    op = str(meta.get("operation", "llm_call"))
    attempt = str(meta.get("attempt", "x"))
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{ts}_{task_id}_{stage}_{op}_a{attempt}_{secrets.token_hex(3)}")

    prompt_path = trace_dir / f"{base}.prompt.txt"
    max_prompt_chars = _env_int("ALPHAOPT_TRACE_MAX_PROMPT_CHARS", 4000, 0)
    prompt_for_storage, prompt_truncated = _truncate_text_for_storage(prompt_text, max_prompt_chars)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_for_storage)

    response_path = None
    response_truncated = False
    if response_text is not None:
        response_path = trace_dir / f"{base}.response.txt"
        max_response_chars = _env_int("ALPHAOPT_TRACE_MAX_RESPONSE_CHARS", 4000, 0)
        response_for_storage, response_truncated = _truncate_text_for_storage(response_text, max_response_chars)
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(response_for_storage)

    return {
        "prompt_id": prompt_path.name,
        "response_id": response_path.name if response_path else None,
        "prompt_truncated": prompt_truncated,
        "response_truncated": response_truncated,
    }


def _write_otlp_friendly_span(
    *,
    trace_output_path: str,
    span_name: str,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    start_ns: int,
    end_ns: int,
    ok: bool,
    attributes: dict | None = None,
    events: list | None = None,
) -> str:
    """
    Write one local OTLP-friendly span record as a JSON line.
    This shape maps directly onto common OTLP span fields.
    """
    trace_dir = Path(trace_output_path) / "_llm_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    use_gzip = _env_bool("ALPHAOPT_TRACE_SPANS_GZIP", False)
    out_path = trace_dir / ("otlp_spans.jsonl.gz" if use_gzip else "otlp_spans.jsonl")
    resource_attrs = {
        "service.name": "alphaopt",
        "service.namespace": "alphaopt",
        "service.version": str(os.getenv("ALPHAOPT_RELEASE", "dev")),
        "deployment.environment": str(os.getenv("ALPHAOPT_ENV", "local")),
        "telemetry.sdk.language": "python",
        "process.pid": os.getpid(),
    }
    resource_attrs.update(_get_experiment_metadata())
    resource_attrs = _compact_dict(resource_attrs)

    record = {
        "resource_attributes": resource_attrs,
        "scope": {
            "name": "alphaopt.llm",
            "version": "1.0",
        },
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id or "",
        "name": span_name,
        "kind": "SPAN_KIND_CLIENT",
        "start_time_unix_nano": int(start_ns),
        "end_time_unix_nano": int(end_ns),
        "status": {
            "code": _to_otlp_status_code(ok),
            "message": "" if ok else "llm_call_failed",
        },
        "attributes": _compact_dict(attributes or {}),
        "events": events or [],
    }

    with _TRACE_WRITE_LOCK:
        if use_gzip:
            import gzip
            with gzip.open(out_path, "at", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(out_path)


def call_llm_and_parse_with_retry(
    model: str,
    service: str | None = None,
    prompt: str | None = None,
    parse_fn: Callable[..., Any] = None,
    temperature: float = 0.7,        # sampling temperature
    max_retry: int = 3,
    sleep_sec: float = 2,
    verbose: bool = True,
    log_header: str | None = None,
    error_message: str | None = None,
    trace_context: dict | None = None,
    trace_output_path: str | None = None,
) -> Any:
    """
    Send a chat prompt to OpenAI or Gemini automatically detected by `model`,
    retry on failure, and parse the raw text with `parse_fn`.
    """
    vendor, client = _build_client(model, service)
    call_trace_id = _new_trace_id_hex()
    resolved_trace_output_path, resolved_run_id = _resolve_trace_output_path(trace_output_path)
    inferred_ctx = _extract_trace_context_from_log_header(log_header)
    base_trace_context = {
        **inferred_ctx,
        **(trace_context or {}),
    }
    experiment_meta = _get_experiment_metadata()
    # Langfuse-style conventional fields when available.
    if "session_id" not in base_trace_context and resolved_run_id:
        base_trace_context["session_id"] = resolved_run_id
    if "trace_name" not in base_trace_context:
        base_trace_context["trace_name"] = "alphaopt.llm_call"
    if "observation_type" not in base_trace_context:
        base_trace_context["observation_type"] = "generation"
    if resolved_run_id and "run_id" not in base_trace_context:
        base_trace_context["run_id"] = resolved_run_id

    def _send_request() -> tuple[str, dict]:
        """
        Dispatch the request to the proper SDK and return raw text.
        """
        # OpenAI / OpenRouter call (OpenAI-compatible)
        if vendor in ("openai", "openrouter"):
            msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            # Build parameters dict
            create_params = {
                "model": model,
                "messages": msgs,
                "temperature": temperature
            }
            # Add frequency_penalty for OpenAI vendor only
            if vendor == "openai":
                create_params["frequency_penalty"] = 0.5
            completion = client.chat.completions.create(**create_params)

            # Token usage from Responses API
            usage = getattr(completion, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None

            # Estimate cost from token usage
            cost = _estimate_cost(vendor, model, prompt_tokens, completion_tokens)
            # If OpenRouter additionally provides usage.cost, you can choose to override or log separately
            if vendor == "openrouter" and usage is not None:
                explicit_cost = getattr(usage, "cost", None)
                if explicit_cost is not None:
                    # Prefer explicit cost if available
                    cost = float(explicit_cost)

            _record_usage(
                vendor,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
            )

            call_meta = {
                "provider": vendor,
                "request_model": model,
                "response_model": getattr(completion, "model", None) or model,
                "request_id": getattr(completion, "id", None),
                "usage_prompt_tokens": prompt_tokens,
                "usage_completion_tokens": completion_tokens,
                "usage_total_tokens": total_tokens,
                "usage_cost_usd": cost,
            }

            return completion.choices[0].message.content, call_meta

        # Gemini call
        if vendor == "gemini":
            from google.genai import types  
            # Disable thinking for gemini-2.5-flash
            if model == "gemini-2.5-flash":               
                completion = client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                    ),
                )
            # gemini-2.5-pro cannot disable thinking
            else:                                               
                completion = client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )

            # Prefer usage_metadata from response to get prompt + completion tokens
            usage_meta = getattr(completion, "usage_metadata", None)
            if usage_meta is not None:
                prompt_tok = getattr(usage_meta, "prompt_token_count", None)
                completion_tok = getattr(usage_meta, "candidates_token_count", None)
                total_tok = getattr(usage_meta, "total_token_count", None)
                cost = _estimate_cost("gemini", model, prompt_tok, completion_tok)
                _record_usage(
                    "gemini",
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                    total_tokens=total_tok,
                    cost=cost,
                )
            else:
                # Fallback: approximate total tokens via models.count_tokens (prompt only)
                try:
                    cost_fallback = 0.0
                    tokens = client.models.count_tokens(model=model, contents=prompt)
                    total_tok = getattr(tokens, "total_tokens", None)
                    if total_tok is not None:
                        cost_fallback = _estimate_cost("gemini", model, total_tok, 0.0)
                        _record_usage(
                            "gemini",
                            prompt_tokens=total_tok,
                            total_tokens=total_tok,
                            cost=cost_fallback,
                        )
                except Exception:
                    # Counting is best-effort; do not fail the main request
                    pass

            prompt_tok = getattr(usage_meta, "prompt_token_count", None) if usage_meta is not None else None
            completion_tok = getattr(usage_meta, "candidates_token_count", None) if usage_meta is not None else None
            total_tok = getattr(usage_meta, "total_token_count", None) if usage_meta is not None else None
            usage_cost = _estimate_cost("gemini", model, prompt_tok, completion_tok)
            call_meta = {
                "provider": vendor,
                "request_model": model,
                "response_model": model,
                "request_id": None,
                "usage_prompt_tokens": prompt_tok,
                "usage_completion_tokens": completion_tok,
                "usage_total_tokens": total_tok,
                "usage_cost_usd": usage_cost,
            }
            return completion.text, call_meta

        raise RuntimeError("Unsupported vendor")

    # Retry loop
    for attempt in range(1, max_retry + 1):
        attempt_start_ns = time.time_ns()
        span_id = _new_span_id_hex()
        prompt_text = _prompt_to_text(prompt)
        prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        parser_name = _callable_name(parse_fn)
        operation_name = str(base_trace_context.get("operation", parser_name))

        if log_header is not None:
            if verbose: print(log_header)
        try:
            t0 = time.time()
            if verbose: print(f"[Attempt {attempt}/{max_retry}]\n")

            raw_text, call_meta = _send_request()
            if verbose: 
                print(raw_text)
            resp_time = time.time() - t0

            if verbose: print(f"Done in {resp_time:.2f}s")

            io_paths = None
            if resolved_trace_output_path and _should_capture_io(on_error=False):
                try:
                    io_paths = _write_llm_io_artifacts(
                        trace_output_path=resolved_trace_output_path,
                        prompt_text=prompt_text,
                        response_text=raw_text,
                        meta={
                            "attempt": attempt,
                            **base_trace_context,
                            "operation": operation_name,
                        },
                    )
                except Exception as io_err:
                    print(f"\n[Tracing warning] failed to persist LLM I/O artifacts: {io_err}")

            result = parse_fn(raw_text)
            if resolved_trace_output_path and _should_write_span(ok=True):
                try:
                    _write_otlp_friendly_span(
                        trace_output_path=resolved_trace_output_path,
                        span_name="llm.call",
                        trace_id=call_trace_id,
                        span_id=span_id,
                        parent_span_id=None,
                        start_ns=attempt_start_ns,
                        end_ns=time.time_ns(),
                        ok=True,
                        attributes={
                            "model_name": model,
                            "service": str(service),
                            "provider": vendor,
                            "llm.provider": call_meta.get("provider"),
                            "llm.request.model": call_meta.get("request_model"),
                            "llm.response.model": call_meta.get("response_model"),
                            "llm.request.id": call_meta.get("request_id"),
                            "temperature": temperature,
                            "attempt": attempt,
                            "max_retry": max_retry,
                            "operation": operation_name,
                            "parser.name": parser_name,
                            "latency_ms": round(resp_time * 1000.0, 3),
                            "level": "DEFAULT",
                            "status_message": "ok",
                            "prompt_sha256": prompt_sha,
                            "prompt_chars": len(prompt_text),
                            "response_chars": len(raw_text or ""),
                            "usage.prompt_tokens": call_meta.get("usage_prompt_tokens"),
                            "usage.completion_tokens": call_meta.get("usage_completion_tokens"),
                            "usage.total_tokens": call_meta.get("usage_total_tokens"),
                            "usage.cost_usd": call_meta.get("usage_cost_usd"),
                            "observation.model": call_meta.get("response_model") or model,
                            "observation.input_id": (io_paths or {}).get("prompt_id"),
                            "observation.output_id": (io_paths or {}).get("response_id"),
                            "prompt_id": (io_paths or {}).get("prompt_id"),
                            "response_id": (io_paths or {}).get("response_id"),
                            **base_trace_context,
                        },
                        events=[],
                    )
                except Exception as span_err:
                    print(f"\n[Tracing warning] failed to persist OTLP span: {span_err}")
            return result

        except Exception as err:
            err_info = _classify_llm_error(err)
            trace_files: dict | None = None
            span_path = None
            io_paths = None
            if resolved_trace_output_path:
                try:
                    if _should_capture_io(on_error=True):
                        io_paths = _write_llm_io_artifacts(
                            trace_output_path=resolved_trace_output_path,
                            prompt_text=prompt_text,
                            response_text=None,
                            meta={
                                "attempt": attempt,
                                **base_trace_context,
                                "operation": operation_name,
                            },
                        )
                except Exception as io_err:
                    print(f"\n[Tracing warning] failed to persist LLM I/O artifacts: {io_err}")

                payload = {
                    "attempt": attempt,
                    "max_retry": max_retry,
                    "model": model,
                    "service": service,
                    "temperature": temperature,
                    "vendor": vendor,
                    "provider": vendor,
                    "operation": operation_name,
                    "parser_name": parser_name,
                    "error_type": err_info["kind"],
                    "status_code": err_info["status_code"],
                    "is_content_filter": err_info["is_content_filter"],
                    "content_filter_result": err_info["content_filter_result"],
                    "error_repr": err_info["error_repr"],
                }
                if io_paths:
                    payload["prompt_id"] = io_paths.get("prompt_id")
                payload.update(base_trace_context)
                if _env_bool("ALPHAOPT_TRACE_ERROR_JSON_ENABLED", True):
                    try:
                        trace_files = _write_llm_error_trace(
                            trace_output_path=resolved_trace_output_path,
                            prompt_text=prompt_text,
                            payload=payload,
                        )
                    except Exception as trace_err:
                        print(f"\n[Tracing warning] failed to persist LLM trace: {trace_err}")

                if _should_write_span(ok=False):
                    try:
                        err_msg_limit = _env_int("ALPHAOPT_TRACE_ERROR_MESSAGE_MAX_CHARS", 1000, 0)
                        err_msg, _ = _truncate_text_for_storage(str(err), err_msg_limit)
                        span_events = [{
                            "name": "exception",
                            "time_unix_nano": time.time_ns(),
                            "attributes": {
                                "exception.type": err.__class__.__name__,
                                "exception.message": err_msg,
                            },
                        }]
                        span_path = _write_otlp_friendly_span(
                            trace_output_path=resolved_trace_output_path,
                            span_name="llm.call",
                            trace_id=call_trace_id,
                            span_id=span_id,
                            parent_span_id=None,
                            start_ns=attempt_start_ns,
                            end_ns=time.time_ns(),
                            ok=False,
                            attributes={
                                "model_name": model,
                                "service": str(service),
                                "provider": vendor,
                                "temperature": temperature,
                                "attempt": attempt,
                                "max_retry": max_retry,
                                "operation": operation_name,
                                "parser.name": parser_name,
                                "level": "ERROR",
                                "status_message": "error",
                                "prompt_sha256": prompt_sha,
                                "prompt_chars": len(prompt_text),
                                "prompt_id": (io_paths or {}).get("prompt_id"),
                                "error_type": err_info["kind"],
                                "status_code": err_info["status_code"],
                                "is_content_filter": err_info["is_content_filter"],
                                "content_filter_result": json.dumps(err_info["content_filter_result"], ensure_ascii=False),
                                **base_trace_context,
                            },
                            events=span_events,
                        )
                    except Exception as span_err:
                        print(f"\n[Tracing warning] failed to persist OTLP span: {span_err}")

            trace_hint = ""
            if trace_files and trace_files.get("trace_json_id"):
                trace_hint = f"\nTrace ID: {trace_files['trace_json_id']}"
            if span_path:
                trace_hint += "\nSpan: written to local OTLP stream"

            # final attempt → raise
            if attempt == max_retry:
                detail = (
                    f"\nLLM request failed after {max_retry} attempts"
                    f"\nType: {err_info['kind']}"
                    f"\nStatus: {err_info['status_code']}"
                    f"{trace_hint}"
                )
                if err_info["is_content_filter"]:
                    detail += f"\nContent filter details: {err_info['content_filter_result']}"
                raise RuntimeError((error_message or detail) + detail if error_message else detail) from err
            # exponential back-off
            backoff = sleep_sec * (2 ** (attempt - 1))
            print(
                f"\nLLM call failed on attempt {attempt}/{max_retry}. "
                f"type={err_info['kind']} status={err_info['status_code']}. "
                f"\nError: {err}.{trace_hint}\nRetrying in {backoff:.1f}s …"
            )
            time.sleep(backoff)

def save_log_data(data, data_path):
    # Save and run corrected code
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    _, ext = os.path.splitext(data_path)
    if data:
        if ext == ".json":
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        if ext == ".txt":
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(data)

        if ext == ".py":
            with open(data_path, "w") as f:
                f.write(data)
    

def cal_time_cost(start_time, phase_name):
    """
    Calculate the the duration of a phase in minutes
    """
    total_minutes = (time.time() - start_time) / 60.0
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    print(f"\n[{phase_name}] took {hours}h {minutes}min")
    return round(total_minutes, 3)


def extract_json_object(text: str):
    """
    Extract the first JSON *object* from an LLM output and return it as a Python dict
    """
    candidate = None
    try:
        # Keep original for debugging
        raw = text

        # Locate the outermost JSON object
        start = raw.find('{')
        end   = raw.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in the text.")
        candidate = raw[start:end+1]

        cand = candidate.strip()

        # Remove trailing commas before ']' or '}'
        cand = re.sub(r",\s*(\]|\})", r"\1", cand)

        # 使用与extract_json_array相同的sanitize_json_like方法
        # 这样可以保持一致性，并且更简洁
        cand = sanitize_json_like(cand)

        # Parse JSON
        result = json.loads(cand)
        if not isinstance(result, dict):
            raise ValueError(f"The parsed JSON is not an object (dict); got {type(result).__name__}")
        return result

    except Exception as e:
        print("LLM raw text:\n", text)
        print("Extracted JSON candidate:\n", candidate if candidate is not None else '<no candidate>')
        print("Error during extracting json object:", repr(e))
        raise


def sanitize_json_like(text: str) -> str:
    # Escape backslashes that are not followed by a valid escape char: " \ / b f n r t u
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Remove trailing commas like ", ]" or ", }"
    text = re.sub(r',\s*([\]\}])', r'\1', text)
    return text


def _extract_json_fence_scope(text: str):
    """
    Return the inner content of a ```json fenced block, but DO NOT close the fence
    if the closing ``` appears inside a JSON string. This prevents early cutoff
    by lines like ```latex that live inside a JSON string value.
    Returns the 'scope' string or None if no json fence exists.
    """
    # Find the opening fence line: ^```[ \t]*json...
    open_pat = re.compile(r'^```[ \t]*json[^\n]*\n', re.IGNORECASE | re.MULTILINE)
    m = open_pat.search(text)
    if not m:
        return None

    i = m.end()                   # start scanning after opening fence newline
    n = len(text)
    in_str = False               # inside a JSON string? (double quotes only)
    escape = False               # previous char was a backslash
    line_start = i               # index of current line start

    while i < n:
        ch = text[i]

        if ch == '\n':
            # track line start
            line_start = i + 1

        if in_str:
            if escape:
                escape = False
            else:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
        else:
            # not in a JSON string
            if ch == '"':
                in_str = True
            else:
                # Only consider a closing fence if it is at the start of a line
                # AND we're not in a JSON string.
                if i == line_start and text.startswith('```', i):
                    # Found the real closing fence
                    return text[m.end():i]

        i += 1

    # No closing fence found; treat until EOF as scope
    return text[m.end():]


def _find_array_slice_bracket_scan(s: str, start_idx: int = None):
    """
    Bracket-aware scan for a top-level JSON array slice in string s.
    Ignores brackets inside JSON strings (double-quoted) and handles escapes.
    Returns (start, end) indices or (None, None).
    """
    n = len(s)
    i = 0 if start_idx is None else max(0, start_idx)
    start = s.find('[', i)
    if start == -1:
        return (None, None)

    in_str = False
    escape = False
    depth = 0
    for j in range(start, n):
        ch = s[j]
        if in_str:
            if escape:
                escape = False
            else:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
            continue
        # not in string
        if ch == '"':
            in_str = True
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            if depth > 0:
                depth -= 1
                if depth == 0:
                    return (start, j)
    return (None, None)


def extract_json_array(text: str):
    """
    Extract the first JSON array from the LLM output and return it as a Python list of dicts.
    - If a ```json fence exists: use a fence-aware scope (won't close on ``` inside JSON strings),
    then bracket-scan to cut the top-level [ ... ].
    - Else: bracket-scan the whole text.
    """
    try:
        # 1) Fence-aware scope extraction
        scope = _extract_json_fence_scope(text)
        if scope is None:
            scope = text  # no json fence; fall back to whole text

        # 2) Bracket-aware slice to get the array block
        start, end = _find_array_slice_bracket_scan(scope)
        if start is None or end is None or end <= start:
            # Coarse fallback (very rare)
            start = scope.find('[')
            end   = scope.rfind(']')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON array found in the text.")
        block = scope[start:end+1].strip()

        # 3) Try direct JSON parsing (raw first)
        try:
            result = json.loads(block)
            if isinstance(result, list) and all(isinstance(x, dict) for x in result):
                return result
        except json.JSONDecodeError:
            pass

        # 4) Try after gentle sanitization
        block2 = sanitize_json_like(block)
        try:
            result = json.loads(block2)
            if isinstance(result, list) and all(isinstance(x, dict) for x in result):
                return result
        except json.JSONDecodeError:
            pass

        # 5) Last resort: scan subsequent top-level arrays (avoid naive regex that hits `[t]`)
        idx = end + 1
        last_fragment = None
        while True:
            s2, e2 = _find_array_slice_bracket_scan(scope, start_idx=idx)
            if s2 is None or e2 is None:
                break
            frag = scope[s2:e2+1].strip()
            last_fragment = frag
            for cand in (frag, sanitize_json_like(frag)):
                try:
                    res = json.loads(cand)
                    if isinstance(res, list) and all(isinstance(x, dict) for x in res):
                        return res
                except Exception:
                    continue
            idx = e2 + 1

        raise ValueError("No valid JSON array of objects found.")

    except Exception as e:
        # Debug output
        print("LLM raw text:\n", text)
        if 'block' in locals():
            print("Extracted block (raw):\n", block[:1000])
        if 'block2' in locals():
            print("Extracted block (sanitized):\n", block2[:1000])
        if 'last_fragment' in locals() and last_fragment is not None:
            print("Last scanned fragment (prefix):\n", last_fragment[:1000])
        print("Error:", repr(e))
        raise
