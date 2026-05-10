import json
from types import SimpleNamespace

import src.llm_programmer as llm_programmer
import src.utils as utils
from src.agent_tracing import agent_step, current_agent_context, record_artifact, record_event


def test_agent_step_writes_local_markdown_and_inherits_context(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAOPT_LAMINAR_ENABLED", "0")
    monkeypatch.setenv("ALPHAOPT_AGENT_TRACE_ENABLED", "1")

    task = SimpleNamespace(id="task-1", ground_truth=10)
    with agent_step(
        "alphaopt.test.step",
        agent_name="Tester",
        operation="unit",
        task=task,
        dataset="demo",
        stage="Program",
        output_path=tmp_path,
    ) as outer:
        with agent_step("ExecuteProgram", agent_name="Solver", operation="execute_code", span_type="TOOL") as inner:
            context = current_agent_context()
            assert context["task_id"] == "task-1"
            assert context["dataset"] == "demo"
            assert context["stage"] == "Program"
            record_artifact("generated_program", "print(1)", artifact_type="code", language="python")
            record_event("program_executed", {"runnable": True})
            inner.set_output({"runnable": True})
        outer.set_output({"status": "ok"})

    trace_root = tmp_path / "_agent_traces"
    assert (trace_root / "agent_steps.jsonl").exists()
    trace_md = (trace_root / "trace.md").read_text()
    assert "```python" in trace_md
    assert "```json" in trace_md

    events = [json.loads(line) for line in (trace_root / "agent_events.jsonl").read_text().splitlines()]
    assert events[-1]["attributes"]["task_id"] == "task-1"
    assert events[-1]["attributes"]["dataset"] == "demo"


def test_record_artifact_deduplicates_payload_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAOPT_LAMINAR_ENABLED", "0")
    monkeypatch.setenv("ALPHAOPT_AGENT_TRACE_ENABLED", "1")

    with agent_step("alphaopt.test.step", output_path=tmp_path):
        first = record_artifact("generated_program", "print(1)", artifact_type="code", language="python")
        second = record_artifact("solver_input_program", "print(1)", artifact_type="code", language="python")

    assert first["path"] == second["path"]
    assert second["duplicate_of"] == first["path"]
    assert len(list((tmp_path / "_agent_traces" / "artifacts").glob("*.py"))) == 1


def test_generate_program_records_non_null_execution_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAOPT_LAMINAR_ENABLED", "0")
    monkeypatch.setenv("ALPHAOPT_AGENT_TRACE_ENABLED", "1")

    def fake_llm_call(**kwargs):
        assert kwargs["trace_context"]["task_id"] == "p1"
        return 'print("Optimal value: 7")'

    def fake_execute_code(self, program):
        assert "Optimal value" in program
        return 7.0

    monkeypatch.setattr(llm_programmer, "call_llm_and_parse_with_retry", fake_llm_call)
    monkeypatch.setattr(llm_programmer.ProgramGenerator, "execute_code", fake_execute_code)

    generator = llm_programmer.ProgramGenerator(model="gpt-test", service="null", temperature=0)
    task = SimpleNamespace(id="p1", desc="demo problem", ground_truth=7.0)
    ablation = SimpleNamespace(include_example=False)

    program, output, runnable, timeout = generator.generate_program(
        iter=1,
        task=task,
        retrieved_insights=[],
        formulation="min x",
        abl_params=ablation,
        verbose=False,
        save_data=False,
        output_path=str(tmp_path),
    )

    assert program
    assert output == 7.0
    assert runnable is True
    assert timeout is False

    trace_md = (tmp_path / "_agent_traces" / "trace.md").read_text()
    assert "Generated Program Execution" in trace_md
    assert "```python" in trace_md
    assert '"alphaopt.ground_truth_objective": 7.0' in trace_md
    assert not (tmp_path / "_llm_traces").exists()


def test_llm_wrapper_creates_agent_step_when_callsite_forgets(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAOPT_LAMINAR_ENABLED", "0")
    monkeypatch.setenv("ALPHAOPT_AGENT_TRACE_ENABLED", "1")

    class FakeCompletion:
        id = "fake-response"
        model = "gpt-test"
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        choices = [SimpleNamespace(message=SimpleNamespace(content="answer"))]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["reasoning_effort"] == "medium"
            return FakeCompletion()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(utils, "_build_client", lambda model, service: ("openai", fake_client))

    result = utils.call_llm_and_parse_with_retry(
        model="gpt-test",
        service="http://127.0.0.1:4010/v1",
        prompt="hello",
        parse_fn=lambda text: {"parsed": text},
        verbose=False,
        trace_output_path=str(tmp_path),
        trace_context={"operation": "forgotten_wrapper", "task_id": "T1", "dataset": "demo", "stage": "Test"},
    )

    assert result == {"parsed": "answer"}
    trace_root = tmp_path / "_agent_traces"
    assert (trace_root / "agent_steps.jsonl").exists()
    assert (trace_root / "agent_artifacts.jsonl").exists()
    assert not (tmp_path / "_llm_traces").exists()
    assert not list(tmp_path.glob("*/_agent_traces"))


def test_no_implicit_testing_traces_without_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALPHAOPT_TRACE_DIR", raising=False)
    monkeypatch.setenv("ALPHAOPT_LAMINAR_ENABLED", "0")
    monkeypatch.setenv("ALPHAOPT_AGENT_TRACE_ENABLED", "1")

    record_event("orphan_event", {"ok": True})

    assert not (tmp_path / "testing" / "traces").exists()
    assert not (tmp_path / "traces").exists()
