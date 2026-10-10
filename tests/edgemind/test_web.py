from types import SimpleNamespace

from edgemind.web import run_provider_probe


class FakeCompletions:
    def __init__(self, text="DeepSeek is ready"):
        self.text = text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))])


class FakeClient:
    def __init__(self, text="DeepSeek is ready"):
        self.completions = FakeCompletions(text)
        self.chat = SimpleNamespace(completions=self.completions)


def test_provider_probe_uses_configured_deepseek_model_and_never_returns_key():
    client = FakeClient()
    result = run_provider_probe("say hello", {"DEEPSEEK_API_KEY": "secret-test-value"}, lambda settings: client)
    assert result["ok"] is True
    assert result["model"] == "deepseek-flash"
    assert result["reply"] == "DeepSeek is ready"
    assert "secret-test-value" not in str(result)
    assert client.completions.kwargs["model"] == "deepseek-flash"


def test_provider_probe_reports_missing_key_without_calling_client():
    called = []
    result = run_provider_probe("test", {}, lambda settings: called.append(settings))
    assert result["ok"] is False
    assert result["error"] == "DEEPSEEK_API_KEY is required for text generation"
    assert called == []


def test_gradio_demo_builds_without_importing_legacy_rag_module():
    import sys
    from edgemind.web import build_demo

    build_demo()
    assert "myrag" not in sys.modules

from edgemind.web import (
    check_builtin_tool_health,
    format_answer_with_sources,
    format_health_result,
    skill_choices,
    tool_readiness_status,
)


def test_tool_readiness_reports_brave_missing_without_secrets():
    status = tool_readiness_status({}, mcp_tools=[])

    by_id = {item["id"]: item for item in status}
    assert by_id["open_meteo"]["ready"] is True
    assert by_id["brave_search"]["ready"] is False
    assert "BRAVE_SEARCH_API_KEY" in by_id["brave_search"]["detail"]
    assert "secret" not in str(status).lower()
    assert "X-Subscription-Token" not in str(status)


def test_tool_readiness_reports_brave_ready_without_echoing_key():
    key = "brave-secret-test"
    status = tool_readiness_status({"BRAVE_SEARCH_API_KEY": key}, mcp_tools=[])

    assert next(item for item in status if item["id"] == "brave_search")["ready"] is True
    assert key not in str(status)


def test_skill_choices_include_general_and_built_in_workflows():
    choices = skill_choices()

    assert choices[0] == ("普通对话（自动工具）", "")
    assert {value for _, value in choices} == {"", "weather_briefing", "web_research"}


def test_manual_health_check_formats_bounded_safe_result():
    class Search:
        def search(self, query, count=1):
            assert count == 1
            return {"status": "succeeded", "result_count": 1}

    result = check_builtin_tool_health("brave_search", {"BRAVE_SEARCH_API_KEY": "secret"},
                                      search_client=Search())
    display = format_health_result(result)

    assert result["ok"] is True
    assert result["result_count"] == 1
    assert "secret" not in str(result)
    assert "正常" in display


def test_manual_health_check_does_not_make_request_when_key_missing():
    class Search:
        def search(self, *args, **kwargs):
            raise AssertionError("must not make request")

    result = check_builtin_tool_health("brave_search", {}, search_client=Search())

    assert result["ok"] is False
    assert result["error"] == "not_configured"


def test_answer_appends_clickable_tool_sources_once():
    answer = format_answer_with_sources("总结内容", {
        "tool_calls": [
            {"source_urls": ["https://example.org/a", "https://example.org/a"]},
            {"source_urls": ["https://docs.example.net/b"]},
        ]
    })

    assert "[example.org](https://example.org/a)" in answer
    assert "[docs.example.net](https://docs.example.net/b)" in answer
    assert answer.count("https://example.org/a") == 1

def test_weather_health_check_counts_location_disambiguation_as_reachable():
    class Weather:
        def get_weather(self, location):
            return {"status": "needs_clarification", "message": "multiple cities"}

    result = check_builtin_tool_health("open_meteo", {}, weather_client=Weather())

    assert result["ok"] is True
    assert result["response_status"] == "needs_clarification"
    assert "多个候选" in result["detail"]


def test_quality_feedback_callback_records_only_run_linked_feedback(tmp_path, monkeypatch):
    import edgemind.web as web
    from edgemind.harness.quality import QualityService
    from edgemind.harness.store import HarnessStore

    store = HarnessStore(tmp_path / "state.sqlite3")
    task = store.create_task("Quality", "Collect outcome feedback")
    from edgemind.harness.models import MemoryItem
    store.put_memory(MemoryItem("mem-1", "semantic", "Known fact", "approved", task.task_id,
                                None, 1.0, {}))
    store.append_event(task.task_id, "agent_turn", {
        "run_id": "run-ui-1", "trace": {"memory_hits": [{"memory_id": "mem-1"}]},
    }, run_id="run-ui-1")
    monkeypatch.setattr(web, "_HARNESS_SERVICE", {"store": store, "quality": QualityService(store)})

    result = web.record_agent_feedback(task.task_id, "run-ui-1", "partial", "memory",
                                       '{"mem-1":"not_helpful"}', "missing important memory")

    assert result.get("outcome") == "partial", result
    assert result["memory_feedback"] == {"mem-1": "not_helpful"}
    assert store.list_agent_feedback(task_id=task.task_id)[0]["run_id"] == "run-ui-1"


def test_candidate_callback_validates_json_and_creates_versioned_policy(tmp_path, monkeypatch):
    import edgemind.web as web
    from edgemind.harness.policies import AgentPolicyService
    from edgemind.harness.store import HarnessStore

    store = HarnessStore(tmp_path / "state.sqlite3")
    policies = AgentPolicyService(store)
    monkeypatch.setattr(web, "_HARNESS_SERVICE", {"store": store, "policies": policies})

    result = web.create_agent_policy_candidate('{"prompt_suffix":"Check evidence before answering."}',
                                               "Improve evidence grounding")

    assert result["ok"] is True
    assert result["policy"]["version"] == 2
    assert result["policy"]["parent_policy_id"] == "baseline-v1"
    assert policies.get(result["policy"]["policy_id"]).prompt_suffix == "Check evidence before answering."
    bad = web.create_agent_policy_candidate('{"python":"print(1)"}', "bad")
    assert bad["ok"] is False


def test_registered_policy_catalog_never_launches_mcp_discovery(tmp_path, monkeypatch):
    import edgemind.web as web
    from edgemind.harness.mcp_client import McpClientManager
    from edgemind.harness.tools import ToolRegistry

    monkeypatch.setattr(web, "_HARNESS_SERVICE", None)
    monkeypatch.setattr(web, "DEFAULT_STATE", tmp_path / "state.sqlite3")
    monkeypatch.setattr(McpClientManager, "configs", lambda self: {
        "ops": {"enabled": True, "tools": ["write_note"]},
    })
    discoveries = []
    monkeypatch.setattr(ToolRegistry, "model_tools", lambda self: discoveries.append("started") or [])

    policy = web._harness_service()["policies"].get_active()

    assert "ops__write_note" in policy.tool_allowlist
    assert discoveries == []


def test_policy_dropdown_choices_include_new_candidate_after_refresh(tmp_path, monkeypatch):
    import edgemind.web as web
    from edgemind.harness.policies import AgentPolicyService
    from edgemind.harness.store import HarnessStore

    store = HarnessStore(tmp_path / "state.sqlite3")
    policies = AgentPolicyService(store)
    monkeypatch.setattr(web, "_HARNESS_SERVICE", {"store": store, "policies": policies})
    candidate = policies.create_candidate({"prompt_suffix": "New"}, "candidate")

    choices, selected = web.agent_policy_dropdown_updates(candidate.policy_id)

    assert ("baseline-v1" in {value for _, value in choices["choices"]})
    assert selected["value"] == candidate.policy_id


def test_fixture_evaluation_callback_persists_comparison_without_live_provider(tmp_path, monkeypatch):
    import edgemind.web as web
    from edgemind.harness.policies import AgentPolicyService
    from edgemind.harness.quality import QualityService
    from edgemind.harness.store import HarnessStore

    store = HarnessStore(tmp_path / "state.sqlite3")
    policies = AgentPolicyService(store)
    candidate = policies.create_candidate({"prompt_suffix": "Check evidence before answering."},
                                          "Improve evidence grounding")
    monkeypatch.setattr(web, "_HARNESS_SERVICE", {"store": store, "policies": policies,
                                                   "quality": QualityService(store)})

    report = web.run_agent_experiment(candidate.policy_id, "tests/data/agent_eval_v1.jsonl",
                                      "fixture", 1, False)

    assert report["passed"] is True, report
    assert report["baseline"]["provider"] == "fixture"
    assert report["candidate"]["provider"] == "fixture"
    assert report["candidate"]["not_applicable_case_count"] == 1
    assert report["candidate"]["applicable_case_count"] == 6
    assert next(case for case in report["candidate"]["case_results"]
                if case["example_id"] == "mcp_confirmation")["passed"] is None
    assert store.get_agent_experiment(report["experiment_id"])["passed"] is True
    assert web.promote_agent_experiment(report["experiment_id"], False)["ok"] is False


def test_gradio_demo_exposes_quality_page_and_feedback_controls():
    from edgemind.web import build_demo

    demo = build_demo()
    labels = [getattr(item, "label", None) for item in demo.blocks.values()]
    values = [getattr(item, "value", None) for item in demo.blocks.values()]

    assert any(item and "Agent 质量" in str(item) for item in values)
    assert any(item and "结果反馈" in item for item in labels)
