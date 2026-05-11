from src.resume_state import (
    make_refinement_result,
    make_training_task_result,
    repair_evaluation_state,
    repair_training_state,
)


class DummyTask:
    def __init__(self, task_id, output_status=None):
        self.id = task_id
        self.output_status = output_status or []

    def to_dict(self, mode="learn"):
        return {
            "task_id": self.id,
            "output_status": self.output_status,
            "retrieved_insights": [],
        }


def test_evaluation_repair_preserves_provider_policy_blocked_records():
    state = {
        "completed_tasks": {
            "blocked": {
                "result": {"status": "provider_policy_blocked"},
                "trace": {"attempts": [], "error": "blocked"},
            },
            "bad": {
                "result": {"status": "experiment_error"},
                "trace": {"attempts": [], "error": "boom"},
            },
        }
    }

    repaired = repair_evaluation_state(state)

    assert repaired == ["bad"]
    assert "blocked" in state["completed_tasks"]
    assert "bad" not in state["completed_tasks"]


def test_training_repair_preserves_provider_policy_and_repairs_experiment_error():
    state = {
        "online_learning": {
            "next_batch_start": 2,
            "completed_task_ids": ["blocked", "bad"],
            "task_results": {
                "blocked": make_training_task_result(
                    task=DummyTask("blocked", ["provider_policy_blocked"]),
                    status="provider_policy_blocked",
                ),
                "bad": make_training_task_result(
                    task=DummyTask("bad", ["experiment_error"]),
                    status="experiment_error",
                    error="boom",
                ),
            },
        },
        "diagnosis": {
            "1": {
                "completed_task_ids": ["blocked", "bad"],
                "task_results": {
                    "blocked": make_training_task_result(
                        task=DummyTask("blocked", ["provider_policy_blocked"]),
                        status="provider_policy_blocked",
                    ),
                    "bad": make_training_task_result(
                        task=DummyTask("bad", ["experiment_error"]),
                        status="experiment_error",
                        error="boom",
                    ),
                },
            }
        },
        "refinement": {
            "1": {
                "completed_insight_ids": ["blocked-ins", "bad-ins"],
                "insight_results": {
                    "blocked-ins": make_refinement_result(
                        insight_id="blocked-ins",
                        status="provider_policy_blocked",
                    ),
                    "bad-ins": make_refinement_result(
                        insight_id="bad-ins",
                        status="experiment_error",
                        error="boom",
                    ),
                },
            }
        },
    }

    repaired = set(repair_training_state(state, tasks=[DummyTask("blocked"), DummyTask("bad")], batch_size=2))

    assert {"bad", "bad-ins"}.issubset(repaired)
    assert "blocked" not in repaired
    assert "blocked-ins" not in repaired
    assert "blocked" in state["online_learning"]["task_results"]
    assert "bad" not in state["online_learning"]["task_results"]
    assert "blocked" in state["diagnosis"]["1"]["task_results"]
    assert "bad" not in state["diagnosis"]["1"]["task_results"]
    assert "blocked-ins" in state["refinement"]["1"]["insight_results"]
    assert "bad-ins" not in state["refinement"]["1"]["insight_results"]


def test_online_learning_batch_rewind_ignores_provider_policy_blocked_task_status():
    state = {
        "online_learning": {
            "next_batch_start": 4,
            "completed_task_ids": ["A", "B", "C", "D"],
            "task_results": {
                "A": make_training_task_result(
                    task=DummyTask("A", ["provider_policy_blocked"]),
                    status="provider_policy_blocked",
                )
            },
        }
    }
    tasks = [
        DummyTask("A", ["provider_policy_blocked"]),
        DummyTask("B", []),
        DummyTask("C", ["experiment_error"]),
        DummyTask("D", []),
    ]

    repaired = repair_training_state(state, tasks=tasks, batch_size=2)

    assert repaired == ["C"]
    assert state["online_learning"]["next_batch_start"] == 2
    assert state["online_learning"]["completed_task_ids"] == ["A", "B"]
