import json
import sqlite3

import pytest

from edgemind.harness.quality import QualityService
from edgemind.harness.store import HarnessStore


def _make_v1_database(path):
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE harness_schema (version INTEGER NOT NULL);
            INSERT INTO harness_schema VALUES (1);
            CREATE TABLE harness_tasks (
                task_id TEXT PRIMARY KEY, title TEXT NOT NULL, objective TEXT NOT NULL,
                status TEXT NOT NULL, state_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE harness_events (
                event_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES harness_tasks(task_id),
                run_id TEXT, event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE harness_memories (
                memory_id TEXT PRIMARY KEY, layer TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL,
                source_task_id TEXT, source_run_id TEXT, confidence REAL NOT NULL,
                metadata_json TEXT NOT NULL, embedding_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE harness_tool_approvals (
                proposal_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES harness_tasks(task_id),
                run_id TEXT NOT NULL, server_name TEXT NOT NULL, tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL, side_effecting INTEGER NOT NULL,
                status TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
        """)
        connection.execute("INSERT INTO harness_tasks VALUES ('task-1','Task','Objective','active','{}','now','now')")
        connection.execute("INSERT INTO harness_tasks VALUES ('task-2','Other','Objective','active','{}','now','now')")
        payload = {"trace": {"memory_hits": [{"memory_id": "memory-1", "layer": "semantic"}]}}
        connection.execute("INSERT INTO harness_events VALUES ('event-1','task-1','run-1','agent_turn',?,'now')",
                           (json.dumps(payload),))
        connection.execute("INSERT INTO harness_memories VALUES ('memory-1','semantic','SQLite','approved','task-1','run-1',0.8,'{}',NULL,'now','now')")
        connection.execute("INSERT INTO harness_tool_approvals VALUES ('proposal-1','task-1','run-1','mcp','write_file','{}',1,'pending',NULL,'now','now')")


def test_v1_database_migrates_additively_and_preserves_existing_rows(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    _make_v1_database(database)

    store = HarnessStore(database)

    assert store.schema_version == 4
    assert store.get_task("task-1").title == "Task"
    assert store.list_events("task-1")[0].run_id == "run-1"
    assert store.get_memory("memory-1").content == "SQLite"
    assert store.get_tool_proposal("proposal-1")["status"] == "pending"


def test_feedback_is_recorded_with_run_and_exposed_memory_ids(tmp_path):
    database = tmp_path / "state.sqlite3"
    _make_v1_database(database)
    store = HarnessStore(database)
    quality = QualityService(store)

    recorded = quality.record_feedback(
        "task-1", "run-1", "partial", "memory",
        {"memory-1": "helpful"}, "The memory helped resume the task.",
    )
    rows = quality.list_feedback(task_id="task-1", run_id="run-1")

    assert recorded["feedback_id"]
    assert recorded["created_at"]
    assert rows[0]["outcome"] == "partial"
    assert rows[0]["failure_category"] == "memory"
    assert rows[0]["note"] == "The memory helped resume the task."
    assert rows[0]["memory_feedback"] == {"memory-1": "helpful"}


def test_provider_failure_run_accepts_outcome_feedback(tmp_path):
    database = tmp_path / "failed.sqlite3"
    _make_v1_database(database)
    store = HarnessStore(database)
    store.append_event("task-1", "agent_turn_failed", {
        "error_type": "ConnectionError", "trace": {"memory_hits": []},
    }, run_id="failed-run")
    quality = QualityService(store)

    result = quality.record_feedback("task-1", "failed-run", "failure", "provider/network")

    assert result["outcome"] == "failure"
    assert result["source_event_id"]


@pytest.mark.parametrize("task_id,run_id,outcome,category,memory_feedback,note", [
    ("task-1", "missing", "success", None, {}, ""),
    ("task-2", "run-1", "success", None, {}, ""),
    ("task-1", "run-1", "unknown", None, {}, ""),
    ("task-1", "run-1", [], None, {}, ""),
    ("task-1", "run-1", "success", None, {"memory-1": []}, ""),
    ("task-1", "run-1", "success", "not-a-category", {}, ""),
    ("task-1", "run-1", "success", None, {"not-exposed": "helpful"}, ""),
    ("task-1", "run-1", "success", None, {"memory-1": "positive"}, ""),
    ("task-1", "run-1", "success", None, {}, "x" * 2001),
    ("task-1", "run-1", "success", None, {}, "api_key=sk-12345678901234567890"),
    ("task-1", "run-1", "success", None, {}, "Authorization: Basic dXNlcjpwYXNzd29yZA=="),
])
def test_invalid_feedback_is_rejected_without_partial_rows(
    tmp_path, task_id, run_id, outcome, category, memory_feedback, note
):
    database = tmp_path / "state.sqlite3"
    _make_v1_database(database)
    quality = QualityService(HarnessStore(database))

    with pytest.raises(ValueError):
        quality.record_feedback(task_id, run_id, outcome, category, memory_feedback, note)

    assert quality.list_feedback() == []
