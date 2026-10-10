from types import SimpleNamespace

from edgemind.agent_eval_fixtures import build_evaluation_runner
from edgemind.harness.builtin_tools import BuiltinToolRegistry
from edgemind.harness.memory import MemoryService
from edgemind.harness.store import HarnessStore
from edgemind.harness.tools import ToolRegistry


class _BuiltinCatalog:
    def model_tools(self):
        return [
            {"type": "function", "function": {"name": "edgemind__get_weather", "parameters": {}}},
            {"type": "function", "function": {"name": "edgemind__web_search", "parameters": {}}},
        ]

    def execute(self, name, arguments, allowed_urls=None):
        return {"status": "succeeded"}


class _McpManager:
    def __init__(self):
        self.discoveries = 0
        self.calls = []

    def list_tools(self):
        self.discoveries += 1
        return [{"server_name": "ops", "name": "write_note", "display_name": "write_note",
                 "description": "write", "input_schema": {"type": "object", "properties": {}},
                 "side_effecting": True}]

    def call_tool(self, *args):
        self.calls.append(args)
        raise AssertionError("live evaluation must not execute MCP")


def test_live_agent_is_restricted_to_case_tools_and_uses_configured_rag(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    manager = _McpManager()
    tools = ToolRegistry(store, manager=manager)
    builtin = _BuiltinCatalog()
    rag = object()
    runner = build_evaluation_runner({"store": store, "memory": MemoryService(store),
                                      "tools": tools, "builtin_tools": builtin, "rag_service": rag})
    case = SimpleNamespace(available_tools=("edgemind__get_weather",), fixture={})
    policy = SimpleNamespace(policy_id="baseline-v1", tool_allowlist=("edgemind__get_weather",))

    agent = runner.agent_factory(store, policy, case, None)

    assert agent.rag_service is rag
    assert {item["function"]["name"] for item in agent.builtin_tools.model_tools()} == {"edgemind__get_weather"}
    assert agent.tools.model_tools() == []
    assert manager.calls == []


def test_fixture_agent_uses_only_injected_fixture_adapters(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    manager = _McpManager()
    runner = build_evaluation_runner({"store": store, "memory": MemoryService(store),
                                      "tools": ToolRegistry(store, manager=manager),
                                      "builtin_tools": BuiltinToolRegistry(), "rag_service": None})
    assert runner.temp_store_factory().db_path == ":memory:"
    assert manager.discoveries == 0
    assert manager.calls == []
