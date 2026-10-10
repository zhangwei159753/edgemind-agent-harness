from types import SimpleNamespace

from edgemind.harness.graph import MemoryGraphService
from edgemind.harness.models import MemoryItem
from edgemind.harness.store import HarnessStore
from edgemind.harness.tasks import TaskService
from edgemind.web import format_memory_graph_paths, list_harness_memory_edges, propose_memory_relation, review_memory_relation


def memory(memory_id, status="approved", content=None):
    return MemoryItem(memory_id, "semantic", content or f"Memory content {memory_id}", status,
                      None, None, 0.8, {})


def test_graph_edge_callbacks_fail_closed_and_support_review(tmp_path, monkeypatch):
    import edgemind.web as web

    store = HarnessStore(tmp_path / "ui.sqlite3")
    store.put_memory(memory("mem-a"))
    store.put_memory(memory("mem-b"))
    graph = MemoryGraphService(store)
    monkeypatch.setattr(web, "_HARNESS_SERVICE", {"store": store, "graph": graph})

    missing = propose_memory_relation("mem-a", "missing", "supports", 0.8)
    pending = propose_memory_relation("mem-a", "mem-b", "supports", 0.85, "run-1")
    pending_edges = list_harness_memory_edges("pending")
    approved = review_memory_relation(pending["edge"]["edge_id"], "approved")

    assert missing["ok"] is False
    assert pending["ok"] is True
    assert pending["edge"]["status"] == "pending"
    assert len(pending_edges) == 1
    assert pending_edges[0]["source_memory_id"] == "mem-a"
    assert approved["ok"] is True
    assert approved["edge"]["status"] == "approved"
    assert list_harness_memory_edges("pending") == []


def test_graph_edge_ui_rejects_unapproved_nodes_and_invalid_decisions(tmp_path, monkeypatch):
    import edgemind.web as web

    store = HarnessStore(tmp_path / "ui-guard.sqlite3")
    store.put_memory(memory("pending-node", status="pending"))
    store.put_memory(memory("approved-node"))
    graph = MemoryGraphService(store)
    monkeypatch.setattr(web, "_HARNESS_SERVICE", {"store": store, "graph": graph})

    edge_result = propose_memory_relation("pending-node", "approved-node", "supports", 0.5)
    rejected_approval = review_memory_relation(edge_result["edge"]["edge_id"], "approved")
    invalid = review_memory_relation(edge_result["edge"]["edge_id"], "pending")

    assert edge_result["ok"] is True
    assert rejected_approval["ok"] is False
    assert invalid["ok"] is False
    assert store.get_memory_edge(edge_result["edge"]["edge_id"]).status == "pending"


def test_graph_path_renderer_is_bounded_and_never_includes_memory_content():
    text = format_memory_graph_paths([
        {"seed_memory_id": "mem-a", "related_memory_id": "mem-b", "relation": "supports",
         "confidence": 0.8, "score": 0.42},
        {"seed_memory_id": "mem-c", "related_memory_id": "mem-d", "relation": "contradicts",
         "confidence": 0.9, "score": 0.0, "kind": "conflict", "content": "secret full memory text"},
    ], limit=1)

    assert "mem-a" in text and "mem-b" in text and "supports" in text
    assert "secret full memory text" not in text
    assert "mem-c" not in text


def test_agent_trace_includes_approved_graph_paths(tmp_path):
    store = HarnessStore(tmp_path / "agent-graph.sqlite3")
    task = TaskService(store).create("Graph context", "Use approved connected memories")
    seed = memory("seed", content="approved deployment rollout policy")
    related = memory("related", content="blue green migration requires validation")
    store.put_memory(seed)
    store.put_memory(related)
    graph = MemoryGraphService(store)
    edge = graph.propose("seed", "related", "supports", 0.9, "source-run")
    graph.review(edge.edge_id, "approved")
    message = SimpleNamespace(content="Use the approved deployment policy.", tool_calls=[])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)]))))

    from edgemind.harness.agent import AgentService
    from edgemind.harness.memory import MemoryService
    turn = AgentService(store, client=client, memory=MemoryService(store)).run_turn(
        task.task_id, "deployment rollout policy")

    assert turn.status == "succeeded"
    assert turn.trace["memory_graph_paths"][0]["relation"] == "supports"
    assert "content" not in turn.trace["memory_graph_paths"][0]
