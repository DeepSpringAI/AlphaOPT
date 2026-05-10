import contextlib
import contextvars
import hashlib
import json
import os
import secrets
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .laminar_tracing import (
    add_span_tags,
    current_session_id,
    current_trace_context,
    current_user_id,
    record_event as laminar_record_event,
    record_exception as laminar_record_exception,
    set_span_attributes,
    set_span_output,
    set_trace_session_id,
    trace_context as laminar_trace_context,
    trace_span,
)


_TRACE_LOCK = None
_CURRENT_OUTPUT_PATH: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "alphaopt_trace_output_path", default=None
)
_CURRENT_STEP_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "alphaopt_trace_step_id", default=None
)
_CURRENT_AGENT_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "alphaopt_agent_context", default={}
)


def _lock():
    global _TRACE_LOCK
    if _TRACE_LOCK is None:
        import threading

        _TRACE_LOCK = threading.Lock()
    return _TRACE_LOCK


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "off", "no")


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(min_value, int(str(raw).strip()))
    except Exception:
        return default


def _primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_primitive(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _primitive(v) for k, v in value.items()}
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except Exception:
        return str(value)


def _compact(data: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        out[str(key)] = _primitive(value)
    return out


def _truncate_text(text: Any, max_chars: int) -> tuple[str, bool, str]:
    raw = "" if text is None else str(text)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    if max_chars and len(raw) > max_chars:
        return raw[:max_chars] + f"\n\n[truncated; sha256={digest}; original_chars={len(raw)}]", True, digest
    return raw, False, digest


def _trace_root(output_path: str | Path | None = None) -> Path | None:
    if not _env_bool("ALPHAOPT_AGENT_TRACE_ENABLED", True):
        return None
    context = current_trace_context()
    resolved = output_path or _CURRENT_OUTPUT_PATH.get() or context.get("output_path")
    if resolved:
        return Path(resolved) / "_agent_traces"
    trace_dir = os.getenv("ALPHAOPT_TRACE_DIR")
    if not trace_dir:
        return None
    base = Path(trace_dir)
    run_id = os.getenv("ALPHAOPT_TRACE_RUN_ID") or f"{time.strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}"
    return base / run_id / "_agent_traces"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock():
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_compact(payload), ensure_ascii=False) + "\n")


def _append_markdown(root: Path, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with _lock():
        with (root / "trace.md").open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")


def _find_existing_artifact(root: Path, digest: str) -> dict[str, Any] | None:
    index = root / "agent_artifacts.jsonl"
    if not digest or not index.exists():
        return None
    try:
        with index.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("sha256") == digest and record.get("path"):
                    path = Path(str(record["path"]))
                    if path.exists():
                        return record
    except Exception:
        return None
    return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _agent_attrs(
    *,
    agent_name: str | None = None,
    operation: str | None = None,
    task: Any = None,
    task_id: Any = None,
    dataset: str | None = None,
    stage: str | None = None,
    attempt: int | None = None,
    iteration: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tid = task_id if task_id is not None else getattr(task, "id", None)
    attrs = {
        "alphaopt.agent": agent_name,
        "alphaopt.operation": operation,
        "alphaopt.task_id": str(tid) if tid is not None else None,
        "alphaopt.dataset": dataset,
        "alphaopt.stage": stage,
        "alphaopt.attempt": attempt,
        "alphaopt.iteration": iteration,
        "alphaopt.ground_truth": getattr(task, "ground_truth", None),
    }
    attrs.update(metadata or {})
    return _compact(attrs)


def _merge_context(context: dict[str, Any] | None) -> dict[str, Any]:
    return _compact({**_CURRENT_AGENT_CONTEXT.get(), **(context or {})})


def current_agent_context() -> dict[str, Any]:
    return _merge_context(current_trace_context())


def current_step_id() -> str | None:
    return _CURRENT_STEP_ID.get()


def _default_tags(
    *,
    agent_name: str | None,
    operation: str | None,
    task_id: Any,
    dataset: Any,
    stage: str | None,
    status: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    tags = ["alphaopt"]
    if agent_name:
        tags.extend([str(agent_name), f"agent:{agent_name}"])
    if operation:
        tags.append(f"op:{operation}")
    if dataset:
        tags.append(f"dataset:{dataset}")
    if task_id is not None:
        tags.append(f"task:{task_id}")
    if stage:
        tags.append(f"stage:{stage}")
    if status:
        tags.append(status)
    tags.extend(extra or [])
    seen: set[str] = set()
    return [t for t in tags if t and not (t in seen or seen.add(t))]


def _laminar_display_name(name: str, span_type: str) -> str:
    if span_type == "DEFAULT" and str(name).startswith("alphaopt."):
        return "agent.step"
    return name


def markdown_code_block(content: Any, language: str = "text", max_chars: int = 12000) -> str:
    text, _, _ = _truncate_text(content, max_chars)
    return f"```{language}\n{text}\n```"


def markdown_execution_summary(
    *,
    title: str,
    code: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    result: dict[str, Any] | None = None,
) -> str:
    parts = [f"### {title}"]
    if result:
        parts.append("```json\n" + json.dumps(_compact(result), ensure_ascii=False, indent=2) + "\n```")
    if code is not None:
        parts.extend(["#### Program", markdown_code_block(code, "python")])
    if stdout is not None:
        parts.extend(["#### stdout", markdown_code_block(stdout, "text")])
    if stderr is not None:
        parts.extend(["#### stderr", markdown_code_block(stderr, "text")])
    return "\n\n".join(parts)


@dataclass
class AgentStep:
    step_id: str
    name: str
    output_path: str | None
    span: Any = None
    output: Any = None
    has_output: bool = False

    def set_output(self, output: Any) -> None:
        self.output = _primitive(output)
        self.has_output = True
        set_span_output(self.span, output)
        root = _trace_root(self.output_path)
        if root is not None:
            _append_jsonl(root / "agent_steps.jsonl", {
                "timestamp": _now(),
                "kind": "step_output",
                "step_id": self.step_id,
                "name": self.name,
                "output": _primitive(output),
            })
            if isinstance(output, str):
                body = output
            else:
                body = "```json\n" + json.dumps(_primitive(output), ensure_ascii=False, indent=2) + "\n```"
            _append_markdown(root, f"\n### Step Output: {self.name}\n\n{body}\n")

    def set_attributes(self, attributes: dict[str, Any] | None) -> None:
        set_span_attributes(self.span, attributes)

    def add_tags(self, tags: list[str] | None) -> None:
        add_span_tags(self.span, tags)


@contextlib.contextmanager
def agent_step(
    name: str,
    *,
    agent_name: str | None = None,
    operation: str | None = None,
    task: Any = None,
    task_id: Any = None,
    dataset: str | None = None,
    stage: str | None = None,
    attempt: int | None = None,
    iteration: int | None = None,
    input: Any = None,
    output_path: str | Path | None = None,
    span_type: str = "DEFAULT",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[AgentStep]:
    step_id = f"{time.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
    out_path = str(output_path) if output_path else _CURRENT_OUTPUT_PATH.get()
    root = _trace_root(out_path)
    started_ns = time.time_ns()
    base_attrs = _agent_attrs(
        agent_name=agent_name,
        operation=operation,
        task=task,
        task_id=task_id,
        dataset=dataset,
        stage=stage,
        attempt=attempt,
        iteration=iteration,
        metadata=attributes,
    )
    inherited = _merge_context({})
    resolved_task_id = base_attrs.get("alphaopt.task_id") or inherited.get("task_id") or inherited.get("alphaopt.task_id")
    if dataset is None:
        dataset = inherited.get("dataset") or inherited.get("alphaopt.dataset")
    if stage is None:
        stage = inherited.get("stage") or inherited.get("alphaopt.stage")
    if attempt is None:
        attempt = inherited.get("attempt") or inherited.get("alphaopt.attempt")
    if iteration is None:
        iteration = inherited.get("iteration") or inherited.get("alphaopt.iteration")
    meta = _compact({
        **inherited,
        "agent": agent_name,
        "operation": operation,
        "span_name": name,
        "task_id": resolved_task_id,
        "dataset": dataset,
        "stage": stage,
        "attempt": attempt,
        "iteration": iteration,
        **(metadata or {}),
    })
    base_attrs = _compact({
        **base_attrs,
        "alphaopt.task_id": resolved_task_id,
        "alphaopt.dataset": dataset,
        "alphaopt.stage": stage,
        "alphaopt.attempt": attempt,
        "alphaopt.iteration": iteration,
    })
    resolved_session_id = session_id or current_session_id()
    resolved_user_id = user_id or current_user_id() or os.getenv("ALPHAOPT_TRACE_USER_ID")
    span_tags = _default_tags(
        agent_name=agent_name,
        operation=operation,
        task_id=resolved_task_id,
        dataset=dataset,
        stage=stage,
        extra=tags,
    )
    if session_id:
        set_trace_session_id(session_id)
    if root is not None:
        _append_jsonl(root / "agent_steps.jsonl", {
            "timestamp": _now(),
            "kind": "step_start",
            "step_id": step_id,
            "name": name,
            "input": _primitive(input),
            "metadata": meta,
            "attributes": base_attrs,
        })
        _append_markdown(root, f"\n## {name}\n\n- step_id: `{step_id}`\n- started: `{_now()}`\n")

    token_path = _CURRENT_OUTPUT_PATH.set(out_path)
    token_step = _CURRENT_STEP_ID.set(step_id)
    token_context = _CURRENT_AGENT_CONTEXT.set(meta)
    try:
        with laminar_trace_context(
            metadata=meta,
            tags=span_tags,
            session_id=resolved_session_id,
            user_id=resolved_user_id,
        ):
            with trace_span(
                _laminar_display_name(name, span_type),
                input=input,
                span_type=span_type,
                tags=span_tags,
                session_id=resolved_session_id,
                user_id=resolved_user_id,
                metadata=meta,
                attributes=base_attrs,
            ) as span:
                step = AgentStep(step_id=step_id, name=name, output_path=out_path, span=span)
                try:
                    yield step
                except Exception as exc:
                    laminar_record_exception(span, exc)
                    add_span_tags(span, ["error"])
                    set_span_attributes(span, {"alphaopt.status": "error", "error.type": exc.__class__.__name__})
                    if root is not None:
                        _append_jsonl(root / "agent_steps.jsonl", {
                            "timestamp": _now(),
                            "kind": "step_error",
                            "step_id": step_id,
                            "name": name,
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                    raise
                else:
                    elapsed_ms = round((time.time_ns() - started_ns) / 1_000_000.0, 3)
                    final_output = step.output if step.has_output else {"status": "ok", "elapsed_ms": elapsed_ms}
                    set_span_output(span, final_output)
                    set_span_attributes(span, {"alphaopt.status": "ok", "alphaopt.elapsed_ms": elapsed_ms})
                    add_span_tags(span, ["ok"])
                    if root is not None:
                        _append_jsonl(root / "agent_steps.jsonl", {
                            "timestamp": _now(),
                            "kind": "step_end",
                            "step_id": step_id,
                            "name": name,
                            "elapsed_ms": elapsed_ms,
                            "status": "ok",
                        })
                        _append_markdown(root, f"- finished: `{_now()}`\n- elapsed_ms: `{elapsed_ms}`\n")
    finally:
        _CURRENT_AGENT_CONTEXT.reset(token_context)
        _CURRENT_OUTPUT_PATH.reset(token_path)
        _CURRENT_STEP_ID.reset(token_step)


def record_event(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    output_path: str | Path | None = None,
) -> None:
    attrs = _compact({
        **_merge_context(current_trace_context()),
        "step_id": _CURRENT_STEP_ID.get(),
        **(attributes or {}),
    })
    laminar_record_event(name, attrs)
    root = _trace_root(output_path)
    if root is not None:
        _append_jsonl(root / "agent_events.jsonl", {
            "timestamp": _now(),
            "kind": "event",
            "name": name,
            "attributes": attrs,
        })


def record_artifact(
    name: str,
    content: Any,
    *,
    artifact_type: str = "text",
    language: str | None = None,
    output_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    root = _trace_root(output_path)
    if root is None:
        return {}
    max_len = max_chars if max_chars is not None else _env_int("ALPHAOPT_AGENT_TRACE_MAX_ARTIFACT_CHARS", 16000, 0)
    text, truncated, digest = _truncate_text(content, max_len)
    existing = _find_existing_artifact(root, digest)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))[:120] or "artifact"
    ext = {
        "code": "py",
        "json": "json",
        "markdown": "md",
        "stdout": "txt",
        "stderr": "txt",
        "text": "txt",
    }.get(artifact_type, "txt")
    if existing:
        path = Path(str(existing["path"]))
    else:
        artifact_dir = root / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_name}_{secrets.token_hex(3)}.{ext}"
        path.write_text(text, encoding="utf-8")
    record = {
        "timestamp": _now(),
        "kind": "artifact",
        "step_id": _CURRENT_STEP_ID.get(),
        "name": name,
        "artifact_type": artifact_type,
        "language": language,
        "path": str(path),
        "sha256": digest,
        "truncated": truncated,
        "duplicate_of": existing.get("path") if existing else None,
        "metadata": _compact(metadata),
    }
    _append_jsonl(root / "agent_artifacts.jsonl", record)
    fence = language or ("python" if artifact_type == "code" else "json" if artifact_type == "json" else "text")
    if existing:
        _append_markdown(root, f"\n### Artifact: {name}\n\nDuplicate content of `{path}`; sha256 `{digest}`.\n")
    else:
        _append_markdown(root, f"\n### Artifact: {name}\n\nPath: `{path}`\n\n```{fence}\n{text}\n```\n")
    return record


def objective_attributes(
    *,
    output: Any,
    ground_truth: Any,
    matched: bool | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "alphaopt.output_objective": None,
        "alphaopt.ground_truth_objective": ground_truth,
        "alphaopt.matched_ground_truth": matched,
    }
    try:
        out = float(output)
        gt = float(ground_truth)
        attrs.update({
            "alphaopt.output_objective": out,
            "alphaopt.absolute_error": abs(out - gt),
            "alphaopt.relative_error": abs(out - gt) / abs(gt) if gt != 0 else None,
        })
    except Exception:
        attrs["alphaopt.output_repr"] = str(output)[:1000]
    return _compact(attrs)
