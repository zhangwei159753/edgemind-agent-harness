import json
from types import SimpleNamespace

import pytest

from edgemind.harness.orchestration.models import ExecutorResult, OrchestrationRequest
from edgemind.harness.orchestration import AgentOrchestrator as PublicAgentOrchestrator
from edgemind.harness.orchestration.runner import AgentOrchestrator
from edgemind.harness.store import HarnessStore


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return response(next_response)


def request(task_id, mode="multi"):
    return OrchestrationRequest(
        task_id=task_id,
        run_id="run-1",
        objective="Research and summarize the topic",
        user_input="Search reliable sources and summarize the findings",
        mode=mode,
        memory_hits=({"memory_id": "memory-1", "content": "Approved preference", "score": 0.9},),
        evidence=({"chunk_id": "doc-1", "content": "Source evidence", "filename": "guide.pdf"},),
        allowed_tools=("edgemind__web_search",),
        base_messages=({"role": "system", "content": "system"}, {"role": "user", "content": "request"}),
    )


def plan_json(expected_evidence=("doc-1", "memory-1")):
    return json.dumps({
        "summary": "Find evidence, then synthesize it.",
        "steps": [{
            "step_id": "step-1", "goal": "Answer using evidence", "depends_on": [],
            "expected_evidence": list(expected_evidence), "expected_tools": ["edgemind__web_search"],
            "risk": "low", "estimated_calls": 1,
        }],
    })


def reviewer_json(decision="accept", instruction=""):
    return json.dumps({"decision": decision, "issues": [], "revision_instruction": instruction})


def make_store(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    task = store.create_task("Test", "Research and summarize")
    return store, task


def test_orchestrator_is_exported_from_harness_package():
    assert PublicAgentOrchestrator is AgentOrchestrator


def test_orchestrator_hands_off_planner_researcher_executor_reviewer_in_order(tmp_path):
    store, task = make_store(tmp_path)
    completions = FakeCompletions([plan_json(), reviewer_json()])
    orchestrator = AgentOrchestrator(SimpleNamespace(chat=SimpleNamespace(completions=completions)), store)
    seen = []

    def executor(handoff, messages):
        seen.append((handoff, messages))
        return ExecutorResult("Answer with source evidence", "succeeded", [], {"tool_calls": []})

    result = orchestrator.run(request(task.task_id), executor, {})

    assert result.status == "succeeded"
    assert [item["role"] for item in result.trace["handoffs"]] == ["planner", "researcher", "executor", "reviewer"]
    assert seen[0][0].role == "executor"
    assert seen[0][0].allowed_tools == ("edgemind__web_search",)
    handoffs = result.trace["handoffs"]
    assert next(item for item in handoffs if item["role"] == "planner")["allowed_tools"] == []
    assert next(item for item in handoffs if item["role"] == "researcher")["allowed_tools"] == []
    assert next(item for item in handoffs if item["role"] == "reviewer")["allowed_tools"] == []
    assert set(seen[0][0].evidence_ids) == {"doc-1", "memory-1"}
    assert "step-1" in str(seen[0][1])
    assert result.trace["review_status"] == "accepted"
    assert store.list_events(task.task_id)


def test_orchestrator_falls_back_safely_when_planner_output_is_malformed(tmp_path):
    store, task = make_store(tmp_path)
    completions = FakeCompletions(["not json"])
    orchestrator = AgentOrchestrator(SimpleNamespace(chat=SimpleNamespace(completions=completions)), store)
    seen = []

    def executor(handoff, messages):
        seen.append(handoff)
        return ExecutorResult("Need manual confirmation", "awaiting_tool_approval", [{"proposal_id": "p-1"}], {})

    result = orchestrator.run(request(task.task_id), executor, {})

    assert result.status == "awaiting_tool_approval"
    assert result.trace["fallback_reason"] == "invalid_role_output"
    assert len(seen) == 1 and seen[0].role == "executor"
    assert seen[0].allowed_tools == ("edgemind__web_search",)
    assert result.pending_tool_calls == [{"proposal_id": "p-1"}]


def test_orchestrator_records_provider_error_type_and_falls_back(tmp_path):
    store, task = make_store(tmp_path)
    completions = FakeCompletions([RuntimeError("secret-bearing-provider-message")])
    orchestrator = AgentOrchestrator(SimpleNamespace(chat=SimpleNamespace(completions=completions)), store)
    seen = []

    def executor(handoff, messages):
        seen.append(handoff)
        return ExecutorResult("Single-agent answer", "succeeded", [], {})

    result = orchestrator.run(request(task.task_id), executor, {})

    assert result.answer == "Single-agent answer"
    assert result.trace["fallback_reason"] == "RuntimeError"
    assert "secret-bearing" not in str(result.trace)
    assert len(seen) == 1


def test_orchestrator_respects_role_budget_and_skips_reviewer_when_exhausted(tmp_path):
    store, task = make_store(tmp_path)
    completions = FakeCompletions([plan_json(), reviewer_json()])
    orchestrator = AgentOrchestrator(SimpleNamespace(chat=SimpleNamespace(completions=completions)), store,
                                     role_call_budget=3)
    executor_calls = []

    result = orchestrator.run(request(task.task_id),
                              lambda handoff, messages: (executor_calls.append(handoff) or
                                  ExecutorResult("Answer", "succeeded", [], {})), {})

    assert result.trace["role_calls"] <= 3
    assert len(executor_calls) == 1
    assert result.trace["review_status"] == "skipped_budget"


def test_reviewer_can_trigger_at_most_one_bounded_revision(tmp_path):
    store, task = make_store(tmp_path)
    completions = FakeCompletions([plan_json(), reviewer_json("revise", "Add the missing comparison."),
                                   reviewer_json("revise", "Keep this final note")])
    orchestrator = AgentOrchestrator(SimpleNamespace(chat=SimpleNamespace(completions=completions)), store)
    executions = []

    def executor(handoff, messages):
        executions.append(messages)
        return ExecutorResult(f"Answer {len(executions)}", "succeeded", [], {})

    result = orchestrator.run(request(task.task_id), executor, {})

    assert len(executions) == 2
    assert result.answer == "Answer 2"
    assert result.trace["revision_count"] == 1
    assert result.trace["review_status"] == "revision_limit_reached"
