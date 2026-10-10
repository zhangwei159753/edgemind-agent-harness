import json
from types import SimpleNamespace

from edgemind.harness.agent import AgentService
from edgemind.harness.orchestration import AgentOrchestrator
from edgemind.harness.store import HarnessStore
from edgemind.harness.tasks import TaskService
from edgemind.harness.tools import ToolRegistry


def completion(content=None, calls=None):
    message = SimpleNamespace(content=content, tool_calls=calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return completion(**self.replies.pop(0))


class EmptyMcp:
    def list_tools(self):
        return []

    def call_tool(self, *args):
        raise AssertionError("unexpected MCP execution")


class SideEffectMcp:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [{
            "server_name": "ops", "name": "write_note", "display_name": "write note",
            "description": "write a note", "input_schema": {"type": "object", "properties": {}},
            "side_effecting": True,
        }]

    def call_tool(self, *args):
        self.calls.append(args)
        raise AssertionError("side-effecting MCP call must wait for confirmation")


def plan(expected_tools=()):
    return json.dumps({"summary": "Complete the requested work using bounded steps.", "steps": [{
        "step_id": "step-1", "goal": "Produce an evidence-based answer", "depends_on": [],
        "expected_evidence": [], "expected_tools": list(expected_tools), "risk": "low", "estimated_calls": 1,
    }]})


def review():
    return json.dumps({"decision": "accept", "issues": [], "revision_instruction": ""})


def build_service(tmp_path, client, manager=None):
    store = HarnessStore(tmp_path / "state.sqlite3")
    task = TaskService(store).create("Research task", "Search and summarize a topic")
    tools = ToolRegistry(store, manager=manager or EmptyMcp())
    service = AgentService(store, client=client, tools=tools, orchestrator=AgentOrchestrator(client, store))
    return service, store, task


def test_legacy_run_turn_uses_single_route_for_simple_request(tmp_path):
    client = FakeClient([{"content": "Hello", "calls": []}])
    service, _, task = build_service(tmp_path, client)

    turn = service.run_turn(task.task_id, "你好")

    assert turn.status == "succeeded"
    assert turn.answer == "Hello"
    assert turn.trace["orchestration"]["mode"] == "single"
    assert turn.trace["orchestration"]["route_reason"] == "simple_request"
    assert len(client.requests) == 1


def test_explicit_multi_route_runs_roles_and_exposes_compact_trace(tmp_path):
    client = FakeClient([
        {"content": plan(), "calls": []},
        {"content": "A concise answer", "calls": []},
        {"content": review(), "calls": []},
    ])
    service, _, task = build_service(tmp_path, client)

    turn = service.run_turn(task.task_id, "完成这项研究", orchestration_mode="multi")

    assert turn.status == "succeeded"
    assert turn.trace["orchestration"]["mode"] == "multi"
    assert turn.trace["orchestration"]["review_status"] == "accepted"
    assert [item["role"] for item in turn.trace["orchestration"]["handoffs"]] == [
        "planner", "researcher", "executor", "reviewer"
    ]
    assert turn.trace["orchestration"]["call_count"] == 3
    assert "chain_of_thought" not in json.dumps(turn.trace["orchestration"])


def test_multi_route_preserves_pending_mcp_confirmation_context(tmp_path):
    manager = SideEffectMcp()
    tool_call = SimpleNamespace(
        id="mcp-call-1", type="function",
        function=SimpleNamespace(name="ops__write_note", arguments='{"note":"draft"}'),
    )
    client = FakeClient([
        {"content": plan(["ops__write_note"]), "calls": []},
        {"content": None, "calls": [tool_call]},
    ])
    service, store, task = build_service(tmp_path, client, manager)

    turn = service.run_turn(task.task_id, "先整理，再保存到笔记", orchestration_mode="multi")

    assert turn.status == "awaiting_tool_approval"
    assert len(turn.pending_tool_calls) == 1
    assert manager.calls == []
    event = next(event for event in store.list_events(task.task_id, limit=100)
                 if event.event_type == "agent_turn" and event.run_id == turn.run_id)
    assert event.payload["tool_context"]
    assert event.payload["tool_proposal_calls"] == [{
        "proposal_id": turn.pending_tool_calls[0]["proposal_id"], "tool_call_id": "mcp-call-1",
    }]
    assert turn.trace["orchestration"]["review_status"] == "skipped_executor_status"
