import pytest
import json
from pathlib import Path

from edgemind.harness.graph import MemoryGraphService
from edgemind.harness.memory import MemoryService
from edgemind.harness.models import MemoryEdge, MemoryHit, MemoryItem
from edgemind.harness.store import HarnessStore


def memory(memory_id, content=None, status="approved"):
    return MemoryItem(memory_id, "semantic", content or f"deployment policy memory {memory_id}",
                      status, None, None, 0.8, {})


def edge(source, target, relation="supports", confidence=0.8, status="approved", edge_id=None):
    return MemoryEdge(edge_id or f"{source}-{target}-{relation}", source, target, relation, status,
                      confidence, "run-1", "", "")


def store_with_memories(tmp_path, *items):
    store = HarnessStore(tmp_path / "graph.sqlite3")
    for item in items:
        store.put_memory(item)
    return store


def test_pending_edge_is_excluded_and_approved_edge_expands_only_one_hop(tmp_path):
    store = store_with_memories(tmp_path, memory("seed"), memory("pending-target"),
                                memory("rejected-target"),
                                memory("connected"), memory("second-hop"))
    graph = MemoryGraphService(store)
    graph.propose("seed", "pending-target", "supports", 0.9)
    rejected_edge = graph.propose("seed", "rejected-target", "supports", 0.9)
    graph.review(rejected_edge.edge_id, "rejected")
    graph.propose("seed", "connected", "supports", 0.8)
    approved = next(item for item in store.list_memory_edges("seed", status="pending")
                    if item.target_memory_id == "connected")
    graph.review(approved.edge_id, "approved")
    seed = MemoryHit(store.get_memory("seed"), 1.0, {"lexical": 1.0})
    graph.propose("connected", "second-hop", "supports", 1.0)
    second_edge = store.list_memory_edges("connected", status="pending")[0]
    graph.review(second_edge.edge_id, "approved")

    expanded = graph.expand([seed])

    ids = {hit.item.memory_id for hit in expanded.hits}
    assert ids == {"seed", "connected"}
    assert "pending-target" not in ids and "rejected-target" not in ids and "second-hop" not in ids


@pytest.mark.parametrize("relation,weight", [
    ("supports", 0.9), ("derived_from", 0.8), ("used_in", 0.7), ("similar_to", 0.6),
])
def test_expanded_scores_use_confidence_and_relation_weight(tmp_path, relation, weight):
    store = store_with_memories(tmp_path, memory("seed"), memory("target"))
    store.put_memory_edge(edge("seed", "target", relation, confidence=0.5))
    seed = MemoryHit(store.get_memory("seed"), 0.8, {"lexical": 1.0})

    expanded = MemoryGraphService(store).expand([seed])

    target = next(hit for hit in expanded.hits if hit.item.memory_id == "target")
    assert target.score == pytest.approx(0.8 * 0.5 * weight)
    assert target.explanation["graph_seed_score"] == 0.8
    assert target.explanation["graph_edge_confidence"] == 0.5
    assert target.explanation["graph_relation_weight"] == weight
    assert all(isinstance(value, (int, float)) for value in target.explanation.values())
    assert expanded.paths[0]["relation"] == relation


def test_contradiction_returns_a_conflict_path_without_positive_expansion(tmp_path):
    store = store_with_memories(tmp_path, memory("seed"), memory("opposing"))
    store.put_memory_edge(edge("seed", "opposing", "contradicts", confidence=0.95))
    seed = MemoryHit(store.get_memory("seed"), 0.7, {})

    expanded = MemoryGraphService(store).expand([seed])

    assert [hit.item.memory_id for hit in expanded.hits] == ["seed"]
    assert expanded.paths == [{
        "seed_memory_id": "seed", "source_memory_id": "seed", "target_memory_id": "opposing",
        "related_memory_id": "opposing", "relation": "contradicts", "confidence": 0.95,
        "relation_weight": 0.0, "score": 0.0, "kind": "conflict",
    }]


def test_retrieve_is_unchanged_by_default_and_trace_can_expand_graph(tmp_path):
    store = store_with_memories(tmp_path, memory("seed", "deployment rollout guidance"),
                                memory("linked", "blue green migration steps"))
    store.put_memory_edge(edge("seed", "linked", "supports", confidence=1.0))
    service = MemoryService(store)

    baseline = service.retrieve("deployment rollout guidance", limit=5)
    traced = service.retrieve_with_trace("deployment rollout guidance", limit=5, graph_expansion=False)
    expanded = service.retrieve_with_trace("deployment rollout guidance", limit=5, graph_expansion=True)

    assert [hit.item.memory_id for hit in baseline] == [hit.item.memory_id for hit in traced.hits]
    assert traced.graph_paths == []
    assert {hit.item.memory_id for hit in expanded.hits} == {"seed", "linked"}
    assert expanded.graph_paths[0]["related_memory_id"] == "linked"


def test_graph_expansion_respects_node_cap_on_high_degree_nodes(tmp_path):
    items = [memory(f"mem-{index}") for index in range(25)]
    store = store_with_memories(tmp_path, *items)
    for index in range(1, 25):
        store.put_memory_edge(edge("mem-0", f"mem-{index}", confidence=1.0))
    seed = MemoryHit(store.get_memory("mem-0"), 1.0, {})

    expanded = MemoryGraphService(store).expand([seed], max_nodes=4)

    assert len(expanded.hits) == 4
    assert len(expanded.paths) <= 4
    assert all(path["related_memory_id"] != "mem-0" for path in expanded.paths)


def test_deleted_endpoints_cannot_leave_dangling_graph_hits(tmp_path):
    store = store_with_memories(tmp_path, memory("seed"), memory("deleted"))
    store.put_memory_edge(edge("seed", "deleted"))
    store.delete_memory("deleted")
    seed = MemoryHit(store.get_memory("seed"), 1.0, {})

    expanded = MemoryGraphService(store).expand([seed])

    assert [hit.item.memory_id for hit in expanded.hits] == ["seed"]
    assert expanded.paths == []


def test_graph_expansion_preserves_task_and_memory_tag_filters(tmp_path):
    store = HarnessStore(tmp_path / "scope.sqlite3")
    task_memory = MemoryItem("seed", "semantic", "deployment policy exact query", "approved",
                             "task-1", None, 0.8, {})
    other_task = MemoryItem("other-task", "semantic", "other task context", "approved",
                            "task-2", None, 0.8, {})
    excluded_tag = MemoryItem("private", "semantic", "private context", "approved",
                              "task-1", None, 0.8, {"tags": ["private"]})
    store.put_memory(task_memory)
    store.put_memory(other_task)
    store.put_memory(excluded_tag)
    store.put_memory_edge(edge("seed", "other-task"))
    store.put_memory_edge(edge("seed", "private"))

    result = MemoryService(store).retrieve_with_trace(
        "deployment policy exact query", task_id="task-1", graph_expansion=True,
        excluded_tags=("private",),
    )

    assert [hit.item.memory_id for hit in result.hits] == ["seed"]
    assert result.graph_paths == []


def test_isolated_graph_fixture_expands_only_reviewed_relations(tmp_path):
    fixture_path = Path("tests/data/memory_graph_v1.jsonl")
    records = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["version"] == 1

    for case in records[1:]:
        store = HarnessStore(tmp_path / f"{case['example_id']}.sqlite3")
        seed = memory(case["seed"]["memory_id"], case["seed"]["content"])
        related = memory(case["related"]["memory_id"], case["related"]["content"])
        store.put_memory(seed)
        store.put_memory(related)
        graph = MemoryGraphService(store)
        edge = graph.propose(seed.memory_id, related.memory_id, case["edge"]["relation"],
                             case["edge"]["confidence"], source_run_id="fixture-run")
        if case["edge"]["status"] == "approved":
            graph.review(edge.edge_id, "approved")
        expansion = graph.expand([MemoryHit(seed, 1.0, {})])
        related_ids = [hit.item.memory_id for hit in expansion.hits if hit.item.memory_id != seed.memory_id]

        assert related_ids == case["expected"]["related_memory_ids"]
        assert len(expansion.paths) == case["expected"]["path_count"]
        assert str(tmp_path) in store.db_path
