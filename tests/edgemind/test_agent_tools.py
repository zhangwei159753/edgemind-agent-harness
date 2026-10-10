import json
from types import SimpleNamespace

from edgemind.harness.agent import AgentService
from edgemind.harness.store import HarnessStore
from edgemind.harness.tasks import TaskService
from edgemind.harness.tools import ToolRegistry


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def assistant(content=None, calls=None, usage=None):
    return SimpleNamespace(content=content, tool_calls=calls or [], usage=usage)


class FakeCompletions:
    def __init__(self, messages):
        self.responses = list(messages)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=message.usage)


class FakeClient:
    def __init__(self, messages):
        self.completions = FakeCompletions(messages)
        self.chat = SimpleNamespace(completions=self.completions)


class FakeBuiltinTools:
    def __init__(self):
        self.calls = []

    def model_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "edgemind__get_weather",
                "description": "weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }, {
            "type": "function",
            "function": {
                "name": "edgemind__web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }, {
            "type": "function",
            "function": {
                "name": "edgemind__fetch_search_result",
                "description": "fetch",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
            },
        }]

    def execute(self, name, arguments, allowed_urls=None):
        self.calls.append((name, arguments, set(allowed_urls or set())))
        if name.endswith("web_search"):
            return {"status": "succeeded", "results": [{
                "url": "https://example.org/source", "title": "Source"
            }], "result_count": 1}
        return {"status": "succeeded", "location": {"name": arguments["location"]},
                "data": {"temperature_2m": 20}}


class EmptyMcpManager:
    def list_tools(self):
        return []

    def call_tool(self, server_name, tool_name, arguments):
        raise AssertionError("no MCP tool should be called")


class SideEffectMcpManager:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [{
            "server_name": "ops", "name": "write_note", "display_name": "write note",
            "description": "write a note", "input_schema": {"type": "object", "properties": {}},
            "side_effecting": True,
        }]

    def call_tool(self, server_name, tool_name, arguments):
        self.calls.append((server_name, tool_name, arguments))
        return {"is_error": False, "content": ["saved"], "structured_content": {"saved": True}}


def build_service(tmp_path, client, builtins=None, manager=None):
    store = HarnessStore(tmp_path / "state.sqlite3")
    task = TaskService(store).create("Weather and research", "Answer with evidence")
    service = AgentService(
        store, client=client, builtin_tools=builtins,
        tools=ToolRegistry(store, manager=manager or EmptyMcpManager()),
    )
    return service, store, task


def test_agent_executes_builtin_tool_and_returns_result_to_deepseek(tmp_path):
    client = FakeClient([
        assistant(calls=[tool_call("edgemind__get_weather", {"location": "杭州"})]),
        assistant(content="杭州现在 20°C。"),
    ])
    builtins = FakeBuiltinTools()
    service, _, task = build_service(tmp_path, client, builtins)
    progress = []

    turn = service.run_turn(task.task_id, "杭州天气如何？", on_tool_status=progress.append)

    assert turn.status == "succeeded"
    assert turn.answer == "杭州现在 20°C。"
    assert builtins.calls[0][0:2] == ("edgemind__get_weather", {"location": "杭州"})
    assert client.completions.requests[1]["messages"][-1]["tool_call_id"] == "call-1"
    assert client.completions.requests[1]["messages"][-1]["role"] == "tool"
    assert turn.trace["tool_rounds"] == 1
    assert turn.trace["tool_calls"][0]["status"] == "succeeded"
    assert progress == ["正在调用 get_weather"]


def test_agent_trace_sums_provider_token_usage_when_available(tmp_path):
    client = FakeClient([assistant(content="Hello", usage=SimpleNamespace(
        prompt_tokens=12, completion_tokens=4, total_tokens=16))])
    service, _, task = build_service(tmp_path, client)

    turn = service.run_turn(task.task_id, "Say hello", orchestration_mode="single")

    assert turn.trace["orchestration"]["call_count"] == 1
    assert turn.trace["orchestration"]["token_usage"] == {
        "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


def test_agent_executes_multiple_builtin_calls_from_one_model_response(tmp_path):
    client = FakeClient([
        assistant(calls=[
            tool_call("edgemind__get_weather", {"location": "杭州"}, "weather-1"),
            tool_call("edgemind__web_search", {"query": "杭州周末活动"}, "search-1"),
        ]),
        assistant(content="天气和活动信息已整理。"),
    ])
    builtins = FakeBuiltinTools()
    service, _, task = build_service(tmp_path, client, builtins)

    turn = service.run_turn(task.task_id, "查天气并找活动")

    assert turn.answer == "天气和活动信息已整理。"
    assert len(builtins.calls) == 2
    tool_messages = [message for message in client.completions.requests[1]["messages"]
                     if message.get("role") == "tool"]
    assert {message["tool_call_id"] for message in tool_messages} == {"weather-1", "search-1"}
    assert turn.trace["tool_calls"][1]["source_urls"] == ["https://example.org/source"]


def test_agent_stops_after_three_tool_call_rounds_and_disables_tools_for_final_answer(tmp_path):
    responses = []
    for index in range(3):
        responses.append(assistant(calls=[
            tool_call("edgemind__get_weather", {"location": f"city-{index}"}, f"call-{index}")
        ]))
    responses.append(assistant(content="按已有结果作答。"))
    client = FakeClient(responses)
    service, _, task = build_service(tmp_path, client, FakeBuiltinTools())

    turn = service.run_turn(task.task_id, "多轮工具问题")

    assert turn.answer == "按已有结果作答。"
    assert len(client.completions.requests) == 4
    assert len(client.completions.requests[3]["messages"]) > 2
    assert "tools" not in client.completions.requests[3]
    assert turn.trace["tool_rounds"] == 3


def test_agent_returns_structured_tool_error_to_model_without_losing_turn(tmp_path):
    client = FakeClient([
        assistant(calls=[SimpleNamespace(
            id="bad-1", type="function",
            function=SimpleNamespace(name="edgemind__get_weather", arguments="{bad json"),
        )]),
        assistant(content="地点参数无法解析，请重新描述。"),
    ])
    service, _, task = build_service(tmp_path, client, FakeBuiltinTools())

    turn = service.run_turn(task.task_id, "天气")

    assert turn.status == "succeeded"
    assert turn.answer == "地点参数无法解析，请重新描述。"
    tool_message = client.completions.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "invalid_arguments" in tool_message["content"]
    assert turn.trace["tool_calls"][0]["status"] == "failed"


def test_side_effect_mcp_proposal_waits_for_confirmation_then_resumes_tool_call(tmp_path):
    client = FakeClient([
        assistant(calls=[tool_call("ops__write_note", {"note": "hello"})]),
        assistant(content="备忘已保存。"),
    ])
    manager = SideEffectMcpManager()
    service, store, task = build_service(tmp_path, client, FakeBuiltinTools(), manager)

    turn = service.run_turn(task.task_id, "保存一条备忘")

    assert turn.status == "awaiting_tool_approval"
    assert manager.calls == []
    proposal_id = turn.pending_tool_calls[0]["proposal_id"]
    outcome = service.execute_tool_proposal(proposal_id, confirmed=True)

    assert outcome["status"] == "executed"
    assert outcome["final_answer"] == "备忘已保存。"
    assert manager.calls == [("ops", "write_note", {"note": "hello"})]
    followup = client.completions.requests[1]["messages"]
    assert any(message.get("role") == "tool" and message.get("tool_call_id") == "call-1"
               for message in followup)
    proposal = store.get_tool_proposal(proposal_id)
    assert proposal["status"] == "executed"



def test_mcp_confirmation_resumes_full_tool_context_with_matching_ids(tmp_path):
    client = FakeClient([
        assistant(calls=[
            tool_call("edgemind__get_weather", {"location": "杭州"}, "weather-1"),
            tool_call("ops__write_note", {"note": "hello"}, "mcp-1"),
        ]),
        assistant(content="天气已查询，备忘已保存。"),
    ])
    manager = SideEffectMcpManager()
    service, _, task = build_service(tmp_path, client, FakeBuiltinTools(), manager)

    turn = service.run_turn(task.task_id, "查天气后保存备忘")

    assert turn.status == "awaiting_tool_approval"
    proposal_id = turn.pending_tool_calls[0]["proposal_id"]
    outcome = service.execute_tool_proposal(proposal_id, confirmed=True)

    messages = client.completions.requests[1]["messages"]
    results = {item.get("tool_call_id"): item.get("content")
               for item in messages if item.get("role") == "tool"}
    assert "weather-1" in results
    assert "mcp-1" in results
    assert "temperature_2m" in results["weather-1"]
    assert outcome["final_answer"] == "天气已查询，备忘已保存。"



def test_selected_skill_limits_schemas_and_rejects_out_of_scope_tool(tmp_path):
    client = FakeClient([assistant(content="完成。")])
    builtins = FakeBuiltinTools()
    service, _, task = build_service(tmp_path, client, builtins)

    turn = service.run_turn(task.task_id, "杭州天气", skill_id="weather_briefing")

    names = {tool["function"]["name"] for tool in client.completions.requests[0]["tools"]}
    assert names == {"edgemind__get_weather"}
    assert turn.trace["skill_id"] == "weather_briefing"
    assert turn.trace["skill_version"] == "1.0"

    invalid_client = FakeClient([
        assistant(calls=[tool_call("edgemind__web_search", {"query": "杭州天气"})]),
        assistant(content="无法调用此工作流未授权的工具。"),
    ])
    invalid_service, _, invalid_task = build_service(tmp_path / "invalid", invalid_client, FakeBuiltinTools())
    invalid_turn = invalid_service.run_turn(invalid_task.task_id, "杭州天气", skill_id="weather_briefing")
    assert invalid_turn.trace["tool_calls"][0]["status"] == "failed"
    assert invalid_service.builtin_tools.calls == []


def test_unknown_skill_fails_before_model_request(tmp_path):
    client = FakeClient([assistant(content="should not happen")])
    service, _, task = build_service(tmp_path, client, FakeBuiltinTools())

    turn = service.run_turn(task.task_id, "hello", skill_id="missing-skill")

    assert turn.status == "failed"
    assert not client.completions.requests


def test_web_research_skill_caps_page_fetches_at_three(tmp_path):
    calls = [tool_call("edgemind__fetch_search_result", {"url": "https://example.org/page"},
                       f"fetch-{index}") for index in range(4)]
    client = FakeClient([
        assistant(calls=calls),
        assistant(content="已整理可读取的页面。"),
    ])
    builtins = FakeBuiltinTools()
    service, _, task = build_service(tmp_path, client, builtins)

    turn = service.run_turn(task.task_id, "调研主题", skill_id="web_research")

    assert len(builtins.calls) == 3
    assert turn.trace["tool_calls"][3]["error_type"] == "skill_limit"
    assert turn.trace["tool_calls"][3]["status"] == "failed"


def seed_passed_experiment(store, candidate, experiment_id):
    store.save_agent_experiment({"experiment_id": experiment_id, "candidate_policy_id": candidate.policy_id,
                                 "baseline_policy_id": candidate.parent_policy_id or "baseline-v1",
                                 "dataset_hash": "test-dataset", "mode": "fixture", "passed": True,
                                 "created_at": "2026-10-10T00:00:00+00:00"})


def test_active_policy_controls_prompt_memory_tools_rounds_and_trace(tmp_path):
    from edgemind.harness.policies import AgentPolicyService

    store = HarnessStore(tmp_path / "state.sqlite3")
    task = TaskService(store).create("Policy task", "Test policy application")
    policies = AgentPolicyService(store, registered_tool_ids=lambda: {
        "edgemind__get_weather", "edgemind__web_search", "edgemind__fetch_search_result",
    })
    candidate = policies.create_candidate({
        "prompt_suffix": "Answer with a short evidence summary.",
        "memory_top_k": 2,
        "layer_weights": {"working": 1.0, "episodic": 0.8, "semantic": 1.2, "procedural": 1.0},
        "tool_allowlist": ["edgemind__get_weather"],
        "max_tool_rounds": 1,
    }, "Keep focused workflows concise.")
    seed_passed_experiment(store, candidate, "eval-1")
    policies.promote(candidate.policy_id, "eval-1", confirmed=True)

    class MemorySpy:
        def __init__(self):
            self.kwargs = None
        def retrieve(self, query, **kwargs):
            self.kwargs = kwargs
            return []
        def make_embedding(self, content):
            return None

    memory = MemorySpy()
    client = FakeClient([
        assistant(calls=[tool_call("edgemind__get_weather", {"location": "杭州"})]),
        assistant(content="基于天气结果简要回答。"),
    ])
    service = AgentService(store, client=client, memory=memory,
                           builtin_tools=FakeBuiltinTools(),
                           tools=ToolRegistry(store, manager=EmptyMcpManager()),
                           policy_service=policies)

    turn = service.run_turn(task.task_id, "杭州天气如何？")

    first_request = client.completions.requests[0]
    assert [item["function"]["name"] for item in first_request["tools"]] == ["edgemind__get_weather"]
    assert "Answer with a short evidence summary." in first_request["messages"][0]["content"]
    assert memory.kwargs["limit"] == 2
    assert memory.kwargs["layer_weights"] == dict(candidate.layer_weights)
    assert turn.trace["tool_rounds"] == 1
    assert turn.trace["policy_id"] == candidate.policy_id
    assert turn.trace["policy_version"] == candidate.version


def test_policy_permissions_are_intersected_with_skill_allowlist(tmp_path):
    from edgemind.harness.policies import AgentPolicyService

    store = HarnessStore(tmp_path / "state.sqlite3")
    task = TaskService(store).create("Skill task", "Test skill permission intersection")
    policies = AgentPolicyService(store, registered_tool_ids=lambda: {
        "edgemind__get_weather", "edgemind__web_search", "edgemind__fetch_search_result",
    })
    candidate = policies.create_candidate({
        "tool_allowlist": ["edgemind__get_weather", "edgemind__web_search", "edgemind__fetch_search_result"]
    }, "Allow registered tools for the general assistant.")
    seed_passed_experiment(store, candidate, "eval-2")
    policies.promote(candidate.policy_id, "eval-2", confirmed=True)
    client = FakeClient([assistant(content="完成。")])
    service = AgentService(store, client=client, builtin_tools=FakeBuiltinTools(),
                           tools=ToolRegistry(store, manager=EmptyMcpManager()), policy_service=policies)

    service.run_turn(task.task_id, "杭州天气", skill_id="weather_briefing")

    names = {item["function"]["name"] for item in client.completions.requests[0]["tools"]}
    assert names == {"edgemind__get_weather"}


def test_policy_does_not_bypass_mcp_side_effect_confirmation(tmp_path):
    from edgemind.harness.policies import AgentPolicyService

    store = HarnessStore(tmp_path / "state.sqlite3")
    task = TaskService(store).create("MCP task", "Keep external actions confirmed")
    manager = SideEffectMcpManager()
    policies = AgentPolicyService(store, registered_tool_ids=lambda: {
        "edgemind__get_weather", "ops__write_note",
    })
    candidate = policies.create_candidate({"tool_allowlist": ["ops__write_note"]}, "Test explicit side-effect approval.")
    seed_passed_experiment(store, candidate, "eval-3")
    policies.promote(candidate.policy_id, "eval-3", confirmed=True)
    client = FakeClient([assistant(calls=[tool_call("ops__write_note", {"note": "hello"})])])
    service = AgentService(store, client=client, builtin_tools=FakeBuiltinTools(),
                           tools=ToolRegistry(store, manager=manager), policy_service=policies)

    turn = service.run_turn(task.task_id, "保存备忘")

    assert turn.status == "awaiting_tool_approval"
    assert manager.calls == []
    assert turn.pending_tool_calls[0]["side_effecting"] is True


def test_run_keeps_policy_snapshot_when_active_policy_changes_midrun(tmp_path):
    from edgemind.harness.policies import AgentPolicyService

    store = HarnessStore(tmp_path / "state.sqlite3")
    task = TaskService(store).create("Snapshot task", "Policy is fixed per run")
    registered = {"edgemind__get_weather", "edgemind__web_search", "edgemind__fetch_search_result"}
    policies = AgentPolicyService(store, registered_tool_ids=lambda: registered)
    first = policies.create_candidate({"prompt_suffix": "Use snapshot A."}, "First policy.")
    second = policies.create_candidate({"prompt_suffix": "Use snapshot B."}, "Second policy.")
    seed_passed_experiment(store, first, "eval-a")
    policies.promote(first.policy_id, "eval-a", confirmed=True)

    class SwitchingCompletions(FakeCompletions):
        def create(self, **kwargs):
            if not self.requests:
                seed_passed_experiment(store, second, "eval-b")
                policies.promote(second.policy_id, "eval-b", confirmed=True)
            return super().create(**kwargs)

    client = FakeClient([assistant(content="Answer from snapshot A.")])
    client.completions = SwitchingCompletions(client.completions.responses)
    client.chat.completions = client.completions
    service = AgentService(store, client=client, builtin_tools=FakeBuiltinTools(),
                           tools=ToolRegistry(store, manager=EmptyMcpManager()), policy_service=policies)

    turn = service.run_turn(task.task_id, "Continue the task")

    assert policies.get_active().policy_id == second.policy_id
    assert turn.trace["policy_id"] == first.policy_id
    assert turn.trace["policy_version"] == first.version
    assert "Use snapshot A." in client.completions.requests[0]["messages"][0]["content"]


def test_mcp_confirmation_followup_uses_the_original_run_policy_snapshot(tmp_path):
    from edgemind.harness.policies import AgentPolicyService

    store = HarnessStore(tmp_path / "state.sqlite3")
    task = TaskService(store).create("Snapshot continuation", "Keep one policy for the whole run")
    tools = {"edgemind__get_weather", "ops__write_note"}
    policies = AgentPolicyService(store, registered_tool_ids=lambda: tools)
    original = policies.create_candidate({"prompt_suffix": "Use original policy A.",
                                          "tool_allowlist": ["ops__write_note"]}, "Policy before confirmation.")
    updated = policies.create_candidate({"prompt_suffix": "Use updated policy B.",
                                         "tool_allowlist": ["ops__write_note"]}, "Policy after confirmation.")
    seed_passed_experiment(store, original, "snapshot-a")
    policies.promote(original.policy_id, "snapshot-a", confirmed=True)
    client = FakeClient([assistant(calls=[tool_call("ops__write_note", {"note": "hello"})]),
                         assistant(content="Done.")])
    service = AgentService(store, client=client, builtin_tools=FakeBuiltinTools(),
                           tools=ToolRegistry(store, manager=SideEffectMcpManager()), policy_service=policies)

    turn = service.run_turn(task.task_id, "Save a note")
    seed_passed_experiment(store, updated, "snapshot-b")
    policies.promote(updated.policy_id, "snapshot-b", confirmed=True)
    service.execute_tool_proposal(turn.pending_tool_calls[0]["proposal_id"], confirmed=True)

    followup_system = client.completions.requests[1]["messages"][0]["content"]
    assert "Use original policy A." in followup_system
    assert "Use updated policy B." not in followup_system


def test_failed_run_keeps_completed_tool_trace_for_feedback_and_evaluation(tmp_path):
    client = FakeClient([assistant(calls=[tool_call("edgemind__get_weather", {"location": "Tokyo"})])])
    service, store, task = build_service(tmp_path, client, FakeBuiltinTools())

    turn = service.run_turn(task.task_id, "Check weather")

    assert turn.status == "failed"
    assert turn.trace["tool_calls"][0]["name"] == "edgemind__get_weather"
    assert turn.trace["tool_calls"][0]["status"] == "succeeded"
    event = next(item for item in store.list_events(task.task_id) if item.run_id == turn.run_id)
    assert event.event_type == "agent_turn_failed"
    assert event.payload["trace"]["tool_calls"]
