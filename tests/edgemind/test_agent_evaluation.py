import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from edgemind.agent_evaluation import AgentEvaluationRunner, load_agent_dataset
from edgemind.harness.agent import AgentService
from edgemind.harness.models import MemoryItem
from edgemind.harness.policies import AgentPolicyService
from edgemind.harness.store import HarnessStore
from edgemind.harness.tasks import TaskService
from edgemind.harness.tools import ToolRegistry

DATASET = Path("tests/data/agent_eval_v1.jsonl")
MULTI_DATASET = Path("tests/data/agent_multi_agent_v1.jsonl")


def _call(name, arguments, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(dict(arguments))))


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if response.get("error") == "TimeoutError":
            raise TimeoutError("fixture provider timeout")
        if response.get("error") == "ConnectionError":
            raise ConnectionError("fixture provider unavailable")
        calls = [_call(item["name"], item.get("arguments", {}), item.get("id", f"call-{len(self.requests)}-{i}"))
                 for i, item in enumerate(response.get("tool_calls", []))]
        message = SimpleNamespace(content=response.get("content"), tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _BuiltinTools:
    def __init__(self, case):
        self.case = case
        self.calls = []
    def model_tools(self):
        schemas = {
            "edgemind__get_weather": {"location": {"type": "string"}},
            "edgemind__web_search": {"query": {"type": "string"}},
            "edgemind__fetch_search_result": {"url": {"type": "string"}},
        }
        return [{"type": "function", "function": {"name": name, "description": name,
                 "parameters": {"type": "object", "properties": schemas[name]}}}
                for name in self.case.available_tools if isinstance(name, str) and name in schemas]
    def execute(self, name, arguments, allowed_urls=None):
        self.calls.append(name)
        output = self.case.fixture.get("tool_outputs", {}).get(name, {"status": "succeeded"})
        if output.get("raise_error"):
            raise RuntimeError("fixture tool error")
        return output


class _McpManager:
    def __init__(self, case):
        self.tools = []
        for item in case.available_tools:
            if isinstance(item, Mapping):
                server, name = item["name"].split("__", 1)
                self.tools.append({"server_name": server, "name": name, "display_name": name,
                                   "description": name, "input_schema": {"type": "object", "properties": {}},
                                   "side_effecting": bool(item.get("side_effecting", False))})
    def list_tools(self):
        return self.tools
    def call_tool(self, *args):
        raise AssertionError("fixture replay must not execute MCP")


class _FixedPolicy:
    def __init__(self, policy):
        self.policy = policy
    def get_active(self):
        return self.policy


class _FixtureRag:
    def __init__(self, citations):
        self.citations = citations
    def ask(self, query):
        return SimpleNamespace(status="succeeded", evidence=[], citations=self.citations,
                               corpus_version="fixture-corpus-v1", run_id="fixture-rag-run")


def _fixtures(case):
    responses = case.fixture.get("responses_by_mode", {}).get(case.orchestration_mode,
                                                               case.fixture.get("responses"))
    return {"client": SimpleNamespace(chat=SimpleNamespace(completions=_Completions(responses))),
            "builtin_tools": _BuiltinTools(case), "mcp_manager": _McpManager(case),
            "rag_service": _FixtureRag(case.fixture.get("citations", []))}


def _agent_factory(store, policy, case, fixtures):
    return AgentService(store, client=fixtures["client"], builtin_tools=fixtures["builtin_tools"],
                        tools=ToolRegistry(store, manager=fixtures["mcp_manager"]),
                        rag_service=fixtures["rag_service"], policy_service=_FixedPolicy(policy),
                        orchestration_model_call_budget=case.max_model_calls)


def test_dataset_loads_manifest_and_all_regression_scenarios():
    dataset = load_agent_dataset(DATASET)
    assert dataset.dataset_id == "edgemind-agent-regression"
    assert dataset.version == 1
    assert len(dataset.cases) >= 6
    assert len({case.example_id for case in dataset.cases}) == len(dataset.cases)
    assert dataset.dataset_hash
    mcp_case = next(case for case in dataset.cases if case.example_id == "mcp_confirmation")
    assert "expected_outcome" in mcp_case.checker_ids
    assert all(case.orchestration_mode == "single" for case in dataset.cases)
    assert all(case.expected_plan_id is None and case.max_model_calls == 8 for case in dataset.cases)


def test_multi_agent_fixture_records_selected_plan_roles_and_call_budget():
    store = HarnessStore(":memory:")
    policy = AgentPolicyService(store).get_active()
    runner = AgentEvaluationRunner(_agent_factory, _fixtures, lambda: HarnessStore(":memory:"))

    report = runner.run(MULTI_DATASET, policy, mode="fixture")

    assert report["metrics"]["case_pass_rate"] == 1.0
    assert report["metrics"]["forbidden_tool_attempts"] == 0
    case = report["case_results"][0]
    assert case["orchestration_mode"] == "multi"
    assert case["selected_plan_id"] == "summary-plan"
    assert case["role_sequence"] == ["planner", "researcher", "executor", "reviewer"]
    assert case["model_call_count"] == 3 <= case["max_model_calls"]
    assert report["metrics"]["model_call_count_mean"] == 11 / 6
    assert report["metrics"]["latency_ms_p95"] is not None


def test_loader_rejects_duplicate_ids_dynamic_code_and_secret_values(tmp_path):
    path = tmp_path / "bad.jsonl"
    manifest = {"record_type": "manifest", "dataset_id": "bad", "version": 1, "description": "bad"}
    case = {"record_type": "case", "example_id": "x", "task": {"title": "T", "objective": "O"},
            "input": "hello", "available_tools": [], "memory_fixtures": [], "expected": {}, "fixture": {"responses": [{"content": "ok"}]}, "checker_ids": []}
    path.write_text("\n".join(json.dumps(item) for item in [manifest, case, case]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_agent_dataset(path)
    case["python"] = "print(1)"
    case["example_id"] = "secret"
    case["input"] = "api_key=sk-12345678901234567890"
    path.write_text("\n".join(json.dumps(item) for item in [manifest, case]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_agent_dataset(path)
    case.pop("python")
    case["input"] = "hello"
    manifest["version"] = 2
    path.write_text("\n".join(json.dumps(item) for item in [manifest, case]), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_agent_dataset(path)


def test_fixture_replay_isolated_uses_fixtures_and_reports_na_metrics(tmp_path):
    production = HarnessStore(tmp_path / "production.sqlite3")
    task = production.create_task("Production", "Must remain untouched")
    production.append_event(task.task_id, "user_note", {"note": "existing"})
    before = (len(production.list_tasks()), len(production.list_events(task.task_id)))
    catalog = {"edgemind__get_weather", "edgemind__web_search", "edgemind__fetch_search_result", "ops__write_note"}
    policy = AgentPolicyService(production, registered_tool_ids=lambda: catalog).get_active()
    seen_stores = []
    def temp_store_factory():
        store = HarnessStore(tmp_path / f"eval-{len(seen_stores)}.sqlite3")
        seen_stores.append(store)
        return store

    runner = AgentEvaluationRunner(_agent_factory, _fixtures, temp_store_factory)
    report = runner.run(DATASET, policy, mode="fixture", repetitions=2)

    assert report["mode"] == "fixture"
    assert report["dataset_hash"]
    assert report["policy_hash"]
    assert report["metrics"]["user_feedback_outcome_rate"] is None
    assert report["na_reasons"]["user_feedback_outcome_rate"]
    assert report["metrics"]["memory_precision_at_k"] is not None
    assert report["metrics"]["memory_recall_at_k"] is not None
    assert report["metrics"]["citation_coverage"] is not None
    assert report["metrics"]["case_pass_rate"] == 1.0
    assert len(report["repeated_metric_summary"]["case_pass_rate"]) == 2
    assert len(seen_stores) >= 12
    assert len({id(store) for store in seen_stores}) == len(seen_stores)
    assert (len(production.list_tasks()), len(production.list_events(task.task_id))) == before
    assert any(case["example_id"] == "mcp_confirmation" and case["actual_status"] == "awaiting_tool_approval"
               for case in report["case_results"]), report["case_results"]
    assert all(case["mode"] == "fixture" for case in report["case_results"]), report["case_results"]


def test_live_evaluation_requires_explicit_confirmation(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    policy = AgentPolicyService(store).get_active()
    runner = AgentEvaluationRunner(_agent_factory, _fixtures, lambda: HarnessStore(":memory:"))
    with pytest.raises(ValueError, match="confirmation"):
        runner.run(DATASET, policy, mode="live")


def test_provider_token_usage_is_summed_without_fabricating_missing_usage():
    from edgemind.harness.usage import sum_usage, usage_from_response

    usage_a = usage_from_response(SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=12, completion_tokens=4, total_tokens=16)))
    usage_b = usage_from_response(SimpleNamespace(usage={"prompt_tokens": 5,
                                                        "completion_tokens": 2, "total_tokens": 7}))

    assert sum_usage([usage_a, usage_b]) == {"prompt_tokens": 17, "completion_tokens": 6,
                                             "total_tokens": 23}
    assert usage_from_response(SimpleNamespace()) is None
    assert sum_usage([]) is None


def test_unavailable_expected_tool_case_is_na_instead_of_a_runtime_failure(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    policy = AgentPolicyService(store).get_active()
    runner = AgentEvaluationRunner(_agent_factory, _fixtures, lambda: HarnessStore(":memory:"))

    report = runner.run(DATASET, policy, mode="fixture")

    mcp_case = next(item for item in report["case_results"] if item["example_id"] == "mcp_confirmation")
    assert mcp_case["actual_status"] == "not_applicable"
    assert mcp_case["passed"] is None
    assert "policy" in mcp_case["na_reason"].lower()
    assert report["metrics"]["case_pass_rate"] == 1.0


def test_policy_checker_ids_add_registered_checks_and_unlabeled_checks_are_na(tmp_path):
    from edgemind.agent_evaluation import AgentDatasetCase
    from edgemind.harness.models import AgentTurn

    store = HarnessStore(tmp_path / "checker.sqlite3")
    policies = AgentPolicyService(store)
    baseline = policies.get_active()
    candidate = policies.create_candidate({"checker_ids": ["expected_outcome"]}, "Check outcomes")
    case = AgentDatasetCase("checker-dispatch", {}, "hello", (), (), {},
                            {"task_outcome": "failure"}, ())
    turn = AgentTurn("run", "task", "succeeded", "ok", {"tool_calls": [], "memory_hits": []})

    baseline_result = AgentEvaluationRunner._evaluate_case(case, turn, baseline, "fixture", 1)
    candidate_result = AgentEvaluationRunner._evaluate_case(case, turn, candidate, "fixture", 1)

    assert "task_outcome" not in baseline_result["checks"]
    assert candidate_result["checks"]["task_outcome"] is False


def test_basic_authorization_is_rejected_and_redacted(tmp_path):
    from edgemind.agent_evaluation import _redact

    value = "Authorization: Basic dXNlcjpwYXNzd29yZA=="
    assert "dXNlcjpwYXNzd29yZA==" not in _redact(value)

    path = tmp_path / "authorization.jsonl"
    manifest = {"record_type": "manifest", "dataset_id": "basic-auth", "version": 1, "description": "test"}
    case = {"record_type": "case", "example_id": "x", "task": {"title": "T", "objective": "O"},
            "input": "hello", "available_tools": [], "memory_fixtures": [], "fixture": {"responses": [{"content": "ok"}]},
            "expected": {}, "checker_ids": [], "Authorization": "Basic dXNlcjpwYXNzd29yZA=="}
    path.write_text("\n".join(json.dumps(item) for item in [manifest, case]), encoding="utf-8")
    with pytest.raises(ValueError, match="credential"):
        load_agent_dataset(path)





def test_forbidden_tool_attempt_is_counted_even_if_registered_and_executed(tmp_path):
    from types import SimpleNamespace
    from edgemind.agent_evaluation import AgentDatasetCase

    store = HarnessStore(tmp_path / "state.sqlite3")
    policy = AgentPolicyService(store).get_active()
    case = AgentDatasetCase(
        "context-forbidden", {}, "do not write", (), (), {},
        {"forbidden_tool_names": ("edgemind__get_weather",)}, ("allowlist",),
    )
    turn = SimpleNamespace(
        status="succeeded", answer="done",
        trace={"tool_calls": [{"name": "edgemind__get_weather", "status": "succeeded"}],
               "memory_hits": [], "elapsed_ms": 5, "model": "fixture"},
    )

    result = AgentEvaluationRunner._evaluate_case(case, turn, policy, "fixture", 1)

    assert result["passed"] is False
    assert result["unsafe_tool_attempts"] == 1
