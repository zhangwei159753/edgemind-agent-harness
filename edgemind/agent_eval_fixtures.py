"""Deterministic provider/tool adapters used only by isolated fixture evaluations."""
from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace

from .agent_evaluation import AgentEvaluationRunner
from .harness.agent import AgentService
from .harness.memory import MemoryService
from .harness.store import HarnessStore
from .harness.tools import ToolRegistry


def _call(name, arguments, call_id):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(dict(arguments))))


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)

    def create(self, **kwargs):
        response = self.responses.pop(0)
        calls = [_call(item["name"], item.get("arguments", {}), item.get("id", f"call-{index}"))
                 for index, item in enumerate(response.get("tool_calls", []))]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=response.get("content"), tool_calls=calls))])


class _BuiltinTools:
    def __init__(self, case):
        self.case = case

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
        raise AssertionError("fixture replay must never execute MCP")


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


class _RestrictedBuiltinTools:
    def __init__(self, registry, allowed_names):
        self.registry = registry
        self.allowed_names = allowed_names

    def model_tools(self):
        return [item for item in self.registry.model_tools()
                if item.get("function", {}).get("name") in self.allowed_names]

    def execute(self, name, arguments, allowed_urls=None):
        if name not in self.allowed_names:
            raise ValueError("tool is outside the live evaluation case allowlist")
        return self.registry.execute(name, arguments, allowed_urls=allowed_urls)


class _RestrictedMcpManager:
    def __init__(self, manager, allowed_names):
        self.manager = manager
        self.allowed_names = allowed_names

    def list_tools(self):
        if not self.allowed_names:
            return []
        return [item for item in self.manager.list_tools()
                if f"{item['server_name']}__{item['name']}" in self.allowed_names]

    def call_tool(self, *args):
        raise RuntimeError("Agent evaluation never executes MCP calls")


def _fixture_factory(case):
    responses = case.fixture.get("responses_by_mode", {}).get(case.orchestration_mode,
                                                               case.fixture.get("responses"))
    return {"client": SimpleNamespace(chat=SimpleNamespace(completions=_Completions(responses))),
            "builtin_tools": _BuiltinTools(case), "mcp_manager": _McpManager(case),
            "rag_service": _FixtureRag(case.fixture.get("citations", []))}


def _fixture_agent_factory(store, policy, case, fixtures):
    if fixtures is not None:
        return AgentService(store, client=fixtures["client"], builtin_tools=fixtures["builtin_tools"],
                            tools=ToolRegistry(store, manager=fixtures["mcp_manager"]),
                            rag_service=fixtures["rag_service"], policy_service=_FixedPolicy(policy),
                            orchestration_model_call_budget=case.max_model_calls)
    return AgentService(store, memory=MemoryService(store), tools=ToolRegistry(store),
                            rag_service=None, policy_service=_FixedPolicy(policy),
                            orchestration_model_call_budget=case.max_model_calls)


def build_evaluation_runner(services) -> AgentEvaluationRunner:
    """Return a runner whose fixture mode is isolated from production state and I/O."""
    def agent_factory(store, policy, case, fixtures):
        if fixtures is not None:
            return _fixture_agent_factory(store, policy, case, fixtures)
        allowed = {item if isinstance(item, str) else item.get("name")
                   for item in case.available_tools}
        real_builtins = services["builtin_tools"]
        real_tools = services["tools"]
        return AgentService(
            store, memory=MemoryService(store),
            tools=ToolRegistry(store, manager=_RestrictedMcpManager(real_tools.manager, allowed)),
            builtin_tools=_RestrictedBuiltinTools(real_builtins, allowed),
            rag_service=services.get("rag_service"), policy_service=_FixedPolicy(policy),
            orchestration_model_call_budget=getattr(case, "max_model_calls", 8),
        )

    return AgentEvaluationRunner(agent_factory, _fixture_factory,
                                 temp_store_factory=lambda: HarnessStore(":memory:"))
