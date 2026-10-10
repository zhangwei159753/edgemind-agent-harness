import json
from types import SimpleNamespace

from edgemind.harness.orchestration import AgentOrchestrator, ExecutorResult, OrchestrationRequest
from edgemind.harness.store import HarnessStore
from edgemind.harness.tasks import TaskService


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return response(item)


def step(step_id="step-1", goal="Answer from evidence", expected_tools=()):
    return {"step_id": step_id, "goal": goal, "depends_on": [], "expected_evidence": ["doc-1"],
            "expected_tools": list(expected_tools), "risk": "low", "estimated_calls": 1}


def candidate(candidate_id, goal_coverage, *, goal="Answer from evidence", expected_tools=(), safety_ok=True):
    return {"candidate_id": candidate_id, "steps": [step(goal=goal, expected_tools=expected_tools)],
            "goal_coverage": goal_coverage, "evidence_availability": 1.0, "tool_availability": 1.0,
            "risk_score": 0.0, "safety_ok": safety_ok, "estimated_cost": 0.0}


def request(task_id, allowed_tools=("web_search",), evidence=({"chunk_id": "doc-1", "content": "Evidence"},)):
    return OrchestrationRequest(
        task_id, "run-1", "Compare options", "Compare the alternatives", "multi", (), evidence,
        allowed_tools, ({"role": "system", "content": "System"}, {"role": "user", "content": "Input"}),
    )


def make_orchestrator(tmp_path, responses, **kwargs):
    store = HarnessStore(tmp_path / "tot.sqlite3")
    task = TaskService(store).create("ToT", "Compare options")
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return AgentOrchestrator(client, store, **kwargs), store, task, completions


def accept():
    return json.dumps({"decision": "accept", "issues": [], "revision_instruction": ""})


def test_only_highest_scoring_safe_plan_is_executed_and_trace_is_compact(tmp_path):
    plans = {"candidates": [
        candidate("candidate-long", 0.7, goal="Choose the highest quality option"),
        candidate("candidate-best", 0.95, goal="Compare the sourced options", expected_tools=("web_search",)),
    ]}
    orchestrator, _, task, completions = make_orchestrator(tmp_path, [json.dumps(plans), accept()])
    executed = []

    result = orchestrator.run(request(task.task_id), lambda handoff, messages: (
        executed.append((handoff, messages)) or ExecutorResult("Selected plan answer", "succeeded", [], {}, model_calls=1)
    ), {})

    assert result.status == "succeeded"
    assert len(executed) == 1
    assert executed[0][0].allowed_tools == ("web_search",)
    assert result.trace["selected_candidate_id"] == "candidate-best"
    assert result.trace["call_count"] == 3
    assert any(item["candidate_id"] == "candidate-long"
               for item in result.trace["plan_selection"]["candidate_summaries"])
    assert "private scratchpad" not in json.dumps(result.trace)
    assert len(completions.calls) == 2


def test_all_unsafe_candidates_return_safe_clarification_without_executor(tmp_path):
    plans = {"candidates": [candidate("unsafe", 1.0, safety_ok=False)]}
    orchestrator, _, task, _ = make_orchestrator(tmp_path, [json.dumps(plans)])
    executor_calls = []

    result = orchestrator.run(request(task.task_id), lambda *args: executor_calls.append(args), {})

    assert result.status == "needs_clarification"
    assert result.trace["review_status"] == "no_safe_plan"
    assert result.trace["selected_candidate_id"] is None
    assert executor_calls == []


def test_planner_provider_error_uses_bounded_existing_executor_fallback(tmp_path):
    orchestrator, _, task, _ = make_orchestrator(tmp_path, [TimeoutError("offline")])
    executor_calls = []

    result = orchestrator.run(request(task.task_id), lambda handoff, messages: (
        executor_calls.append((handoff, messages)) or ExecutorResult("Fallback answer", "succeeded", [], {}, model_calls=1)
    ), {})

    assert result.answer == "Fallback answer"
    assert result.trace["fallback_reason"] == "TimeoutError"
    assert len(executor_calls) == 1
    assert executor_calls[0][1] == list(request(task.task_id).base_messages)


def test_model_call_budget_exhaustion_aborts_before_executor(tmp_path):
    plans = {"candidates": [candidate("candidate-best", 0.95)]}
    orchestrator, _, task, completions = make_orchestrator(
        tmp_path, [json.dumps(plans)], model_call_budget=1)
    executor_calls = []

    result = orchestrator.run(request(task.task_id), lambda *args: executor_calls.append(args), {})

    assert result.status == "needs_clarification"
    assert result.trace["review_status"] == "model_call_budget_exhausted"
    assert result.trace["call_count"] == 1
    assert executor_calls == []
    assert len(completions.calls) == 1


def test_malformed_plan_and_disallowed_tool_never_execute_unvalidated_plan(tmp_path):
    malformed = {"candidates": [candidate("bad", 0.9, expected_tools=("not-allowed",))]}
    orchestrator, _, task, _ = make_orchestrator(tmp_path, [json.dumps(malformed)])
    executor_calls = []

    result = orchestrator.run(request(task.task_id), lambda *args: executor_calls.append(args), {})

    assert result.status == "needs_clarification"
    assert result.trace["review_status"] == "no_safe_plan"
    assert executor_calls == []
