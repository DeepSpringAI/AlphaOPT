import contextlib
import contextvars
import hashlib
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse


_INIT_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()
_INITIALIZED = False
_ENABLED = False
_ENV_LOADED = False
_CURRENT_METADATA: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "alphaopt_laminar_metadata", default={}
)
_CURRENT_TAGS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "alphaopt_laminar_tags", default=()
)
_CURRENT_LABELS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "alphaopt_laminar_labels", default=()
)
_CURRENT_SESSION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "alphaopt_laminar_session_id", default=None
)
_CURRENT_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "alphaopt_laminar_user_id", default=None
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "off", "no")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def laminar_enabled() -> bool:
    _load_default_laminar_env()
    return _env_bool("ALPHAOPT_LAMINAR_ENABLED", False)


def _load_default_laminar_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path("infra/.env")
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except Exception:
        pass


def _fail_fast_enabled() -> bool:
    return _env_bool("ALPHAOPT_LAMINAR_FAIL_FAST", True)


def _primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _compact(data: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (data or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and not v:
            continue
        key = getattr(k, "value", k)
        out[str(key)] = _primitive(v)
    return out


def _merge_tags(*tag_groups: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in tag_groups:
        for tag in group or []:
            tag_str = str(tag)
            if not tag_str or tag_str in seen:
                continue
            seen.add(tag_str)
            merged.append(tag_str)
    return merged


@contextlib.contextmanager
def trace_context(
    *,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    labels: list[str] | tuple[str, ...] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Temporarily inherit Laminar context for nested AlphaOPT spans."""
    merged_metadata = _compact({**_CURRENT_METADATA.get(), **(metadata or {})})
    merged_tags = tuple(_merge_tags(_CURRENT_TAGS.get(), tags))
    merged_labels = tuple(_merge_tags(_CURRENT_LABELS.get(), labels))
    resolved_session_id = session_id or _CURRENT_SESSION_ID.get()
    resolved_user_id = user_id or _CURRENT_USER_ID.get()

    token_meta = _CURRENT_METADATA.set(merged_metadata)
    token_tags = _CURRENT_TAGS.set(merged_tags)
    token_labels = _CURRENT_LABELS.set(merged_labels)
    token_session = _CURRENT_SESSION_ID.set(resolved_session_id)
    token_user = _CURRENT_USER_ID.set(resolved_user_id)
    try:
        yield merged_metadata
    finally:
        _CURRENT_METADATA.reset(token_meta)
        _CURRENT_TAGS.reset(token_tags)
        _CURRENT_LABELS.reset(token_labels)
        _CURRENT_SESSION_ID.reset(token_session)
        _CURRENT_USER_ID.reset(token_user)


def current_trace_context() -> dict[str, Any]:
    return dict(_CURRENT_METADATA.get())


def current_session_id() -> str | None:
    return _CURRENT_SESSION_ID.get()


def current_user_id() -> str | None:
    return _CURRENT_USER_ID.get()


def _config_sha256(config_path: str | None) -> str | None:
    if not config_path:
        return None
    p = Path(config_path)
    if not p.exists():
        return None
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _assert_laminar_reachable(base_url: str, http_port: int, grpc_port: int, timeout: float) -> None:
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    host = parsed.hostname
    if not host:
        raise RuntimeError(f"Invalid ALPHAOPT_LAMINAR_BASE_URL: {base_url!r}")
    errors = []
    for label, port in (("http", http_port), ("grpc", grpc_port)):
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                pass
        except OSError as exc:
            errors.append(f"{label} {host}:{port} ({exc})")
    if errors:
        raise RuntimeError("Laminar backend is not reachable: " + "; ".join(errors))


def default_run_metadata(*, mode: str, config_path: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = {
        "mode": mode,
        "config_path": config_path,
        "config_name": Path(config_path).name if config_path else None,
        "config_sha256": _config_sha256(config_path),
        "git_commit": _git_commit(),
    }
    metadata.update(extra or {})
    return _compact(metadata)


def init_laminar_from_env(*, mode: str, config_path: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
    """
    Initialize Laminar once. Disabled mode is a no-op. Enabled mode fails fast by
    default so Laminar-backed experiment commands never silently lose traces.
    """
    global _INITIALIZED, _ENABLED
    if not laminar_enabled():
        _ENABLED = False
        return False

    with _INIT_LOCK:
        if _INITIALIZED:
            return True

        api_key = os.getenv("LMNR_PROJECT_API_KEY") or os.getenv("LAMINAR_PROJECT_API_KEY")
        if not api_key:
            raise RuntimeError("ALPHAOPT_LAMINAR_ENABLED=1 but LMNR_PROJECT_API_KEY is not set.")

        try:
            from lmnr import Laminar
        except Exception as exc:
            raise RuntimeError("ALPHAOPT_LAMINAR_ENABLED=1 but the lmnr package is not importable.") from exc

        base_url = os.getenv("ALPHAOPT_LAMINAR_BASE_URL", "http://localhost")
        http_port = _env_int("ALPHAOPT_LAMINAR_HTTP_PORT", 8000)
        grpc_port = _env_int("ALPHAOPT_LAMINAR_GRPC_PORT", 8001)
        export_timeout = _env_int("ALPHAOPT_LAMINAR_EXPORT_TIMEOUT_SECONDS", 5)
        init_metadata = default_run_metadata(mode=mode, config_path=config_path, extra=metadata)

        try:
            _assert_laminar_reachable(base_url, http_port, grpc_port, timeout=min(float(export_timeout), 5.0))
            Laminar.initialize(
                project_api_key=api_key,
                base_url=base_url,
                http_port=http_port,
                grpc_port=grpc_port,
                instruments=[],
                export_timeout_seconds=export_timeout,
                metadata=init_metadata,
            )
            with Laminar.start_as_current_span(
                "alphaopt.laminar.startup",
                input={"mode": mode, "config_path": config_path},
                tags=["alphaopt", "startup", mode],
                metadata=init_metadata,
                attributes={
                    "alphaopt.mode": mode,
                    "alphaopt.config_path": config_path or "",
                    "alphaopt.laminar.base_url": base_url,
                    "alphaopt.laminar.http_port": http_port,
                    "alphaopt.laminar.grpc_port": grpc_port,
                },
            ) as span:
                span.set_output({"ok": True})
            Laminar.force_flush()
        except Exception as exc:
            if _fail_fast_enabled():
                raise RuntimeError(
                    "Failed to initialize/export to Laminar. Check infra/.env, LMNR_PROJECT_API_KEY, "
                    "and the self-hosted app-server on ports 8000/8001."
                ) from exc
            print(f"[Laminar warning] tracing disabled after initialization failure: {exc}")
            _ENABLED = False
            _INITIALIZED = True
            return False

        _ENABLED = True
        _INITIALIZED = True
        print(f"[Laminar] enabled for {mode}: {base_url}:{http_port} grpc={grpc_port}")
        return True


@contextlib.contextmanager
def trace_span(
    name: str,
    *,
    input: Any = None,
    span_type: str = "DEFAULT",
    tags: list[str] | None = None,
    labels: list[str] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    parent_span_context: Any = None,
) -> Iterator[Any]:
    inherited_metadata = _CURRENT_METADATA.get()
    inherited_tags = _CURRENT_TAGS.get()
    inherited_labels = _CURRENT_LABELS.get()
    metadata_payload = _compact({**inherited_metadata, **(metadata or {})})
    tag_payload = _merge_tags(inherited_tags, tags)
    label_payload = _merge_tags(inherited_labels, labels)
    resolved_session_id = session_id or _CURRENT_SESSION_ID.get()
    resolved_user_id = user_id or _CURRENT_USER_ID.get()

    token_meta = _CURRENT_METADATA.set(metadata_payload)
    token_tags = _CURRENT_TAGS.set(tuple(tag_payload))
    token_labels = _CURRENT_LABELS.set(tuple(label_payload))
    token_session = _CURRENT_SESSION_ID.set(resolved_session_id)
    token_user = _CURRENT_USER_ID.set(resolved_user_id)
    if not laminar_enabled() or not _ENABLED:
        try:
            yield None
        finally:
            _CURRENT_METADATA.reset(token_meta)
            _CURRENT_TAGS.reset(token_tags)
            _CURRENT_LABELS.reset(token_labels)
            _CURRENT_SESSION_ID.reset(token_session)
            _CURRENT_USER_ID.reset(token_user)
        return
    from lmnr import Laminar

    try:
        with Laminar.start_as_current_span(
            name,
            input=input,
            span_type=span_type,
            labels=label_payload or None,
            tags=tag_payload or None,
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            metadata=metadata_payload,
            attributes=_compact(attributes),
            parent_span_context=parent_span_context,
        ) as span:
            yield span
    finally:
        _CURRENT_METADATA.reset(token_meta)
        _CURRENT_TAGS.reset(token_tags)
        _CURRENT_LABELS.reset(token_labels)
        _CURRENT_SESSION_ID.reset(token_session)
        _CURRENT_USER_ID.reset(token_user)


def set_span_output(span: Any, output: Any) -> None:
    if span is None:
        return
    try:
        span.set_output(output)
    except Exception:
        pass


def set_span_attributes(span: Any, attributes: dict[str, Any] | None) -> None:
    if span is None:
        return
    try:
        span.set_attributes(_compact(attributes))
    except Exception:
        pass


def add_span_tags(span: Any, tags: list[str] | None) -> None:
    if span is None or not tags:
        return
    try:
        span.add_tags(tags)
    except Exception:
        pass


def record_exception(span: Any, exc: BaseException) -> None:
    if span is None:
        return
    try:
        span.record_exception(exc)
    except Exception:
        pass


def record_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    if not laminar_enabled() or not _ENABLED:
        return
    try:
        from lmnr import Laminar

        Laminar.event(name, attributes=_compact(attributes))
    except Exception:
        pass


def set_trace_metadata(metadata: dict[str, Any] | None) -> None:
    if not laminar_enabled() or not _ENABLED:
        return
    try:
        from lmnr import Laminar

        Laminar.set_trace_metadata(_compact(metadata))
    except Exception:
        pass


def set_trace_session_id(session_id: str | None) -> None:
    if not session_id or not laminar_enabled() or not _ENABLED:
        return
    try:
        from lmnr import Laminar

        Laminar.set_trace_session_id(str(session_id))
    except Exception:
        pass


def set_trace_user_id(user_id: str | None) -> None:
    if not user_id or not laminar_enabled() or not _ENABLED:
        return
    try:
        from lmnr import Laminar

        Laminar.set_trace_user_id(str(user_id))
    except Exception:
        pass


def current_trace_id() -> str | None:
    if not laminar_enabled() or not _ENABLED:
        return None
    try:
        from lmnr import Laminar

        trace_id = Laminar.get_trace_id()
        return str(trace_id) if trace_id is not None else None
    except Exception:
        return None


def serialize_current_span_context() -> str | None:
    if not laminar_enabled() or not _ENABLED:
        return None
    try:
        from lmnr import Laminar

        return Laminar.serialize_span_context()
    except Exception:
        return None


def deserialize_span_context(payload: str | dict | None) -> Any:
    if not payload or not laminar_enabled() or not _ENABLED:
        return None
    try:
        from lmnr import Laminar

        return Laminar.deserialize_span_context(payload)
    except Exception:
        return None


def llm_attributes(
    *,
    provider: str,
    request_model: str,
    response_model: str | None,
    request_id: str | None,
    prompt_tokens: float | None,
    completion_tokens: float | None,
    total_tokens: float | None,
    cost: float | None,
    input_cost: float | None = None,
    output_cost: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[Any, Any]:
    attrs: dict[Any, Any] = {}
    try:
        from lmnr import Attributes

        attrs.update(
            {
                Attributes.PROVIDER: provider,
                Attributes.REQUEST_MODEL: request_model,
                Attributes.RESPONSE_MODEL: response_model or request_model,
                Attributes.INPUT_TOKEN_COUNT: float(prompt_tokens or 0.0),
                Attributes.OUTPUT_TOKEN_COUNT: float(completion_tokens or 0.0),
                Attributes.TOTAL_TOKEN_COUNT: float(total_tokens or 0.0),
                Attributes.TOTAL_COST: float(cost or 0.0),
                Attributes.INPUT_COST: float(input_cost or 0.0),
                Attributes.OUTPUT_COST: float(output_cost or 0.0),
            }
        )
        if request_id:
            attrs[Attributes.RESPONSE_ID] = request_id
    except Exception:
        attrs.update(
            {
                "gen_ai.system": provider,
                "gen_ai.request.model": request_model,
                "gen_ai.response.model": response_model or request_model,
                "gen_ai.usage.input_tokens": float(prompt_tokens or 0.0),
                "gen_ai.usage.output_tokens": float(completion_tokens or 0.0),
                "llm.usage.total_tokens": float(total_tokens or 0.0),
                "gen_ai.usage.cost": float(cost or 0.0),
                "gen_ai.usage.input_cost": float(input_cost or 0.0),
                "gen_ai.usage.output_cost": float(output_cost or 0.0),
            }
        )
        if request_id:
            attrs["gen_ai.response.id"] = request_id
    attrs.update(_compact(extra))
    return attrs


def record_trace_index(output_dir: str | Path | None, record: dict[str, Any]) -> None:
    if not output_dir or not laminar_enabled() or not _ENABLED:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **_compact(record),
    }
    with _INDEX_LOCK:
        with (out_dir / "laminar_trace_index.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def flush_laminar() -> None:
    if not laminar_enabled() or not _ENABLED:
        return
    try:
        from lmnr import Laminar

        Laminar.force_flush()
    except Exception as exc:
        if _fail_fast_enabled():
            raise RuntimeError("Laminar flush failed.") from exc
        print(f"[Laminar warning] flush failed: {exc}")
