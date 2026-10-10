import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from edgemind.harness.models import MemoryEdge, MemoryItem
from edgemind.harness.quality import QualityService
from edgemind.harness.store import HarnessStore


def memory(memory_id):
    return MemoryItem(memory_id, "semantic", f"Memory {memory_id}", "approved", None, None, 0.8, {})


def edge(source="mem-a", target="mem-b", relation="supports", status="pending", confidence=0.8,
         edge_id="edge-1", source_run_id="run-1"):
    return MemoryEdge(edge_id, source, target, relation, status, confidence, source_run_id, "", "")


def test_v4_database_migrates_to_v5_without_losing_memories_or_feedback(tmp_path):
    db = tmp_path / "legacy-v4.sqlite3"
    store = HarnessStore(db)
    task = store.create_task("Migration", "Preserve existing data")
    store.put_memory(memory("mem-a"))
    event = store.append_event(task.task_id, "agent_turn", {"answer": "kept"}, "run-1")
    QualityService(store).record_feedback(task.task_id, "run-1", "success")
    with store._connect() as conn:
        conn.execute("DROP TABLE harness_memory_edges")
        conn.execute("UPDATE harness_schema SET version=4")

    migrated = HarnessStore(db)

    assert migrated.schema_version == 5
    assert migrated.get_memory("mem-a").content == "Memory mem-a"
    assert migrated.list_agent_feedback(task_id=task.task_id)[0]["outcome"] == "success"
    assert migrated.get_task(task.task_id).objective == "Preserve existing data"


def test_v5_migration_is_idempotent_on_reopen(tmp_path):
    db = tmp_path / "reopen.sqlite3"
    HarnessStore(db)

    reopened = HarnessStore(db)
    reopened_again = HarnessStore(db)

    assert reopened.schema_version == reopened_again.schema_version == 5
    with reopened_again._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM harness_memory_edges").fetchone()[0] == 0


def test_memory_edge_round_trip_review_and_incident_listing(tmp_path):
    store = HarnessStore(tmp_path / "edges.sqlite3")
    store.put_memory(memory("mem-a"))
    store.put_memory(memory("mem-b"))

    created = store.put_memory_edge(edge())

    assert created.status == "pending"
    assert created.confidence == 0.8
    assert store.get_memory_edge("edge-1") == created
    assert store.list_memory_edges("mem-a") == []
    reviewed = store.review_memory_edge("edge-1", "approved")
    assert reviewed.status == "approved"
    assert store.list_memory_edges("mem-a") == [reviewed]
    assert store.list_memory_edges("mem-b") == [reviewed]
    with pytest.raises(FrozenInstanceError):
        reviewed.status = "rejected"


@pytest.mark.parametrize("kwargs", [
    {"relation": "unknown"},
    {"status": "active"},
    {"confidence": -0.1},
    {"confidence": 1.1},
    {"confidence": float("nan")},
    {"source": "mem-a", "target": "mem-a"},
])
def test_invalid_memory_edges_are_rejected(kwargs):
    with pytest.raises(ValueError):
        edge(**kwargs)


def test_duplicate_edges_and_reverse_similar_edges_are_rejected(tmp_path):
    store = HarnessStore(tmp_path / "duplicates.sqlite3")
    store.put_memory(memory("mem-a"))
    store.put_memory(memory("mem-b"))
    store.put_memory_edge(edge(relation="supports", edge_id="support-1"))
    store.put_memory_edge(edge(source="mem-a", target="mem-b", relation="similar_to", edge_id="similar-1"))

    with pytest.raises(ValueError, match="duplicate"):
        store.put_memory_edge(edge(relation="supports", edge_id="support-2"))
    with pytest.raises(ValueError, match="duplicate"):
        store.put_memory_edge(edge(source="mem-b", target="mem-a", relation="similar_to", edge_id="similar-2"))


def test_memory_delete_cascades_edges_and_missing_endpoints_fail_closed(tmp_path):
    store = HarnessStore(tmp_path / "cascade.sqlite3")
    store.put_memory(memory("mem-a"))
    store.put_memory(memory("mem-b"))
    store.put_memory_edge(edge())
    assert store.delete_memory("mem-a") is True
    assert store.get_memory_edge("edge-1") is None
    assert store.delete_memory_edges_for_memory("mem-b") == 0
    with pytest.raises(ValueError, match="memory"):
        store.put_memory_edge(edge(edge_id="dangling"))

