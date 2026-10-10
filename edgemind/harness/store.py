import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import HarnessEvent, MemoryItem, TaskRecord


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class HarnessStore:
    """Additive SQLite persistence isolated from the existing RAG tables."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._dsn = f"file:edgemind-{uuid.uuid4().hex}?mode=memory&cache=shared" if self.db_path == ":memory:" else self.db_path
        self._uri = self.db_path == ":memory:"
        self._keeper = self._connect() if self._uri else None
        self.fts_available = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._dsn, uri=self._uri, timeout=10, factory=_ClosingConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS harness_schema (version INTEGER NOT NULL);
                INSERT INTO harness_schema(version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM harness_schema);
                CREATE TABLE IF NOT EXISTS harness_tasks (
                    task_id TEXT PRIMARY KEY, title TEXT NOT NULL, objective TEXT NOT NULL,
                    status TEXT NOT NULL, state_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS harness_events (
                    event_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES harness_tasks(task_id),
                    run_id TEXT, event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_harness_events_task ON harness_events(task_id, created_at);
                CREATE TABLE IF NOT EXISTS harness_memories (
                    memory_id TEXT PRIMARY KEY, layer TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL,
                    source_task_id TEXT, source_run_id TEXT, confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL, embedding_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_harness_memories_status_layer ON harness_memories(status, layer);
                CREATE TABLE IF NOT EXISTS harness_tool_approvals (
                    proposal_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES harness_tasks(task_id),
                    run_id TEXT NOT NULL, server_name TEXT NOT NULL, tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL, side_effecting INTEGER NOT NULL,
                    status TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
            """)
            try:
                conn.executescript("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS harness_memory_fts USING fts5(memory_id UNINDEXED, content, layer UNINDEXED);
                    CREATE TRIGGER IF NOT EXISTS harness_memory_ai AFTER INSERT ON harness_memories BEGIN
                      INSERT INTO harness_memory_fts(memory_id, content, layer) VALUES (new.memory_id, new.content, new.layer);
                    END;
                    CREATE TRIGGER IF NOT EXISTS harness_memory_ad AFTER DELETE ON harness_memories BEGIN
                      DELETE FROM harness_memory_fts WHERE memory_id = old.memory_id;
                    END;
                    CREATE TRIGGER IF NOT EXISTS harness_memory_au AFTER UPDATE OF content, layer ON harness_memories BEGIN
                      DELETE FROM harness_memory_fts WHERE memory_id = old.memory_id;
                      INSERT INTO harness_memory_fts(memory_id, content, layer) VALUES (new.memory_id, new.content, new.layer);
                    END;
                """)
                conn.execute("""INSERT INTO harness_memory_fts(memory_id, content, layer)
                    SELECT m.memory_id, m.content, m.layer FROM harness_memories m
                    WHERE NOT EXISTS (SELECT 1 FROM harness_memory_fts f WHERE f.memory_id=m.memory_id)""")
                self.fts_available = True
            except sqlite3.OperationalError:
                self.fts_available = False
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT version FROM harness_schema LIMIT 1").fetchone()
            version = int(row[0]) if row else 1
            if version < 2:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS harness_agent_feedback (
                        feedback_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL REFERENCES harness_tasks(task_id),
                        run_id TEXT NOT NULL,
                        source_event_id TEXT NOT NULL REFERENCES harness_events(event_id),
                        outcome TEXT NOT NULL,
                        failure_category TEXT,
                        note TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_harness_agent_feedback_task
                        ON harness_agent_feedback(task_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_harness_agent_feedback_run
                        ON harness_agent_feedback(run_id, created_at);
                    CREATE TABLE IF NOT EXISTS harness_agent_memory_feedback (
                        memory_feedback_id TEXT PRIMARY KEY,
                        feedback_id TEXT NOT NULL REFERENCES harness_agent_feedback(feedback_id) ON DELETE CASCADE,
                        memory_id TEXT NOT NULL REFERENCES harness_memories(memory_id) ON DELETE CASCADE,
                        rating TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(feedback_id, memory_id)
                    );
                    UPDATE harness_schema SET version=2;
                """)
            row = conn.execute("SELECT version FROM harness_schema LIMIT 1").fetchone()
            version = int(row[0]) if row else 1
            if version < 3:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS harness_agent_policies (
                        policy_id TEXT PRIMARY KEY,
                        version INTEGER NOT NULL UNIQUE,
                        parent_policy_id TEXT,
                        payload_json TEXT NOT NULL,
                        rationale TEXT NOT NULL,
                        source_run_id TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS harness_agent_policy_state (
                        singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
                        active_policy_id TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS harness_agent_policy_transitions (
                        transition_id TEXT PRIMARY KEY,
                        action TEXT NOT NULL CHECK(action IN ('promote','rollback')),
                        from_policy_id TEXT NOT NULL,
                        to_policy_id TEXT NOT NULL,
                        experiment_id TEXT,
                        created_at TEXT NOT NULL
                    );
                    UPDATE harness_schema SET version=3;
                """)
            row = conn.execute("SELECT version FROM harness_schema LIMIT 1").fetchone()
            version = int(row[0]) if row else 1
            if version < 4:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS harness_agent_experiments (
                        experiment_id TEXT PRIMARY KEY,
                        candidate_policy_id TEXT NOT NULL,
                        baseline_policy_id TEXT NOT NULL,
                        dataset_hash TEXT NOT NULL,
                        mode TEXT NOT NULL CHECK(mode IN ('fixture','live')),
                        passed INTEGER NOT NULL CHECK(passed IN (0,1)),
                        report_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_harness_agent_experiments_candidate
                        ON harness_agent_experiments(candidate_policy_id, created_at);
                    UPDATE harness_schema SET version=4;
                """)
    @property
    def schema_version(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT version FROM harness_schema LIMIT 1").fetchone()
        return int(row[0])

    def create_task(self, title: str, objective: str, task_id: str | None = None) -> TaskRecord:
        title, objective = title.strip(), objective.strip()
        if not title or not objective:
            raise ValueError("task title and objective are required")
        now, identity = _now(), task_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute("INSERT INTO harness_tasks VALUES (?, ?, ?, 'active', '{}', ?, ?)", (identity, title, objective, now, now))
        return self.get_task(identity)

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM harness_tasks WHERE task_id=?", (task_id,)).fetchone()
        return None if row is None else self._task(row)

    def list_tasks(self, statuses: tuple[str, ...] | None = None, limit: int = 100) -> list[TaskRecord]:
        with self._connect() as conn:
            if statuses:
                marks = ",".join("?" for _ in statuses)
                rows = conn.execute(f"SELECT * FROM harness_tasks WHERE status IN ({marks}) ORDER BY updated_at DESC LIMIT ?", (*statuses, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM harness_tasks ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._task(row) for row in rows]

    def update_task_state(self, task_id: str, status: str | None = None, state: dict[str, Any] | None = None) -> TaskRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM harness_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise ValueError(f"unknown task: {task_id}")
            conn.execute("UPDATE harness_tasks SET status=?, state_json=?, updated_at=? WHERE task_id=?",
                         (status or row["status"], _dump(state if state is not None else json.loads(row["state_json"])), _now(), task_id))
        return self.get_task(task_id)

    def append_event(self, task_id: str, event_type: str, payload: dict[str, Any], run_id: str | None = None) -> HarnessEvent:
        event = HarnessEvent(str(uuid.uuid4()), task_id, run_id, event_type, payload, _now())
        with self._connect() as conn:
            conn.execute("INSERT INTO harness_events VALUES (?, ?, ?, ?, ?, ?)",
                         (event.event_id, task_id, run_id, event_type, _dump(payload), event.created_at))
        return event

    def create_agent_policy(self, policy: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("""INSERT INTO harness_agent_policies
                (policy_id, version, parent_policy_id, payload_json, rationale, source_run_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (policy["policy_id"], policy["version"], policy["parent_policy_id"], _dump(policy["payload"]),
                 policy["rationale"], policy["source_run_id"], policy["created_at"]))

    def get_agent_policy(self, policy_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM harness_agent_policies WHERE policy_id=?", (policy_id,)).fetchone()
        if row is None:
            return None
        return {"policy_id": row["policy_id"], "version": row["version"],
                "parent_policy_id": row["parent_policy_id"], "payload": json.loads(row["payload_json"]),
                "rationale": row["rationale"], "source_run_id": row["source_run_id"], "created_at": row["created_at"]}

    def list_agent_policies(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM harness_agent_policies ORDER BY version DESC").fetchall()
        return [{"policy_id": row["policy_id"], "version": row["version"],
                 "parent_policy_id": row["parent_policy_id"], "payload": json.loads(row["payload_json"]),
                 "rationale": row["rationale"], "source_run_id": row["source_run_id"],
                 "created_at": row["created_at"]} for row in rows]

    def save_agent_experiment(self, report: dict[str, Any]) -> None:
        report_json = _dump(report)
        if len(report_json.encode("utf-8")) > 1_000_000:
            raise ValueError("Agent experiment report exceeds the 1 MB persistence limit")
        with self._connect() as conn:
            conn.execute("""INSERT INTO harness_agent_experiments
                (experiment_id, candidate_policy_id, baseline_policy_id, dataset_hash, mode, passed, report_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (report["experiment_id"], report["candidate_policy_id"], report["baseline_policy_id"],
                 report.get("dataset_hash", ""), report["mode"], int(bool(report.get("passed"))),
                 report_json, report["created_at"]))

    def get_agent_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM harness_agent_experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            return None
        return {"experiment_id": row["experiment_id"], "candidate_policy_id": row["candidate_policy_id"],
                "baseline_policy_id": row["baseline_policy_id"], "dataset_hash": row["dataset_hash"],
                "mode": row["mode"], "passed": bool(row["passed"]), "created_at": row["created_at"],
                "report": json.loads(row["report_json"])}

    def list_agent_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM harness_agent_experiments ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"experiment_id": row["experiment_id"], "candidate_policy_id": row["candidate_policy_id"],
                 "baseline_policy_id": row["baseline_policy_id"], "dataset_hash": row["dataset_hash"],
                 "mode": row["mode"], "passed": bool(row["passed"]), "created_at": row["created_at"],
                 "report": json.loads(row["report_json"])} for row in rows]

    def activate_agent_policy(self, policy_id: str, action: str, experiment_id: str | None = None) -> None:
        if action not in {"promote", "rollback"}:
            raise ValueError("invalid policy transition")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if policy_id != "baseline-v1" and conn.execute(
                    "SELECT 1 FROM harness_agent_policies WHERE policy_id=?", (policy_id,)).fetchone() is None:
                raise ValueError("unknown policy")
            if action == "promote":
                experiment = conn.execute("""SELECT passed, candidate_policy_id FROM harness_agent_experiments
                    WHERE experiment_id=?""", (experiment_id,)).fetchone()
                if experiment is None or not experiment["passed"] or experiment["candidate_policy_id"] != policy_id:
                    raise ValueError("policy promotion requires a matching passed experiment gate")
            row = conn.execute("SELECT active_policy_id FROM harness_agent_policy_state WHERE singleton_id=1").fetchone()
            current = row[0] if row else "baseline-v1"
            if action == "rollback" and policy_id != "baseline-v1":
                previously_active = conn.execute("""SELECT 1 FROM harness_agent_policy_transitions
                    WHERE action='promote' AND to_policy_id=? LIMIT 1""", (policy_id,)).fetchone()
                if previously_active is None:
                    raise ValueError("rollback target must be a previously active policy")
            if row is None:
                conn.execute("INSERT INTO harness_agent_policy_state(singleton_id, active_policy_id, updated_at) VALUES (1, ?, ?)",
                             (policy_id, _now()))
            else:
                conn.execute("UPDATE harness_agent_policy_state SET active_policy_id=?, updated_at=? WHERE singleton_id=1",
                             (policy_id, _now()))
            conn.execute("""INSERT INTO harness_agent_policy_transitions
                (transition_id, action, from_policy_id, to_policy_id, experiment_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), action, current, policy_id, experiment_id, _now()))

    def list_agent_policy_transitions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM harness_agent_policy_transitions ORDER BY created_at").fetchall()
        return [{"transition_id": row["transition_id"], "action": row["action"],
                 "from_policy_id": row["from_policy_id"], "to_policy_id": row["to_policy_id"],
                 "experiment_id": row["experiment_id"], "created_at": row["created_at"]} for row in rows]

    def active_agent_policy_id(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT active_policy_id FROM harness_agent_policy_state WHERE singleton_id=1").fetchone()
        return None if row is None else row[0]
    def find_agent_turn_event(self, run_id: str, task_id: str | None = None) -> HarnessEvent | None:
        query = "SELECT * FROM harness_events WHERE run_id=? AND event_type IN ('agent_turn','agent_turn_failed')"
        params: list[Any] = [run_id]
        if task_id is not None:
            query += " AND task_id=?"
            params.append(task_id)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return HarnessEvent(row["event_id"], row["task_id"], row["run_id"], row["event_type"],
                            json.loads(row["payload_json"]), row["created_at"])

    def insert_agent_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute("""INSERT INTO harness_agent_feedback
                (feedback_id, task_id, run_id, source_event_id, outcome, failure_category, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (feedback["feedback_id"], feedback["task_id"], feedback["run_id"],
                 feedback["source_event_id"], feedback["outcome"], feedback["failure_category"],
                 feedback["note"], feedback["created_at"]))
            conn.executemany("""INSERT INTO harness_agent_memory_feedback
                (memory_feedback_id, feedback_id, memory_id, rating, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                [(entry["memory_feedback_id"], feedback["feedback_id"], entry["memory_id"],
                  entry["rating"], feedback["created_at"]) for entry in feedback["memory_feedback"]])
        return feedback

    def list_agent_feedback(self, task_id: str | None = None, run_id: str | None = None,
                            limit: int = 100) -> list[dict[str, Any]]:
        clauses, params = [], []
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(task_id)
        if run_id is not None:
            clauses.append("run_id=?")
            params.append(run_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM harness_agent_feedback" + where +
                                " ORDER BY created_at DESC LIMIT ?", (*params, limit)).fetchall()
            feedback = []
            for row in rows:
                memories = conn.execute("""SELECT memory_id, rating FROM harness_agent_memory_feedback
                    WHERE feedback_id=? ORDER BY memory_id""", (row["feedback_id"],)).fetchall()
                feedback.append({"feedback_id": row["feedback_id"], "task_id": row["task_id"],
                                 "run_id": row["run_id"], "source_event_id": row["source_event_id"],
                                 "outcome": row["outcome"], "failure_category": row["failure_category"],
                                 "note": row["note"], "created_at": row["created_at"],
                                 "memory_feedback": {item["memory_id"]: item["rating"] for item in memories}})
        return feedback
    def list_events(self, task_id: str, limit: int = 100) -> list[HarnessEvent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM harness_events WHERE task_id=? ORDER BY created_at DESC LIMIT ?", (task_id, limit)).fetchall()
        return [HarnessEvent(r["event_id"], r["task_id"], r["run_id"], r["event_type"], json.loads(r["payload_json"]), r["created_at"]) for r in rows]

    def put_memory(self, item: MemoryItem) -> MemoryItem:
        with self._connect() as conn:
            conn.execute("""INSERT INTO harness_memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET layer=excluded.layer, content=excluded.content,
                status=excluded.status, source_task_id=excluded.source_task_id, source_run_id=excluded.source_run_id,
                confidence=excluded.confidence, metadata_json=excluded.metadata_json,
                embedding_json=excluded.embedding_json, updated_at=excluded.updated_at""",
                (item.memory_id, item.layer, item.content, item.status, item.source_task_id, item.source_run_id,
                 item.confidence, _dump(item.metadata), None if item.embedding is None else _dump(item.embedding),
                 item.created_at or _now(), _now()))
        return self.get_memory(item.memory_id)

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM harness_memories WHERE memory_id=?", (memory_id,)).fetchone()
        return None if row is None else self._memory(row)

    def list_memories(self, status: str | None = None, layer: str | None = None, limit: int = 200) -> list[MemoryItem]:
        clauses, params = [], []
        if status:
            clauses.append("status=?"); params.append(status)
        if layer:
            clauses.append("layer=?"); params.append(layer)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM harness_memories" + where + " ORDER BY updated_at DESC LIMIT ?", (*params, limit)).fetchall()
        return [self._memory(row) for row in rows]

    def review_memory(self, memory_id: str, status: str, content: str | None = None) -> MemoryItem:
        if status not in {"approved", "rejected", "superseded"}:
            raise ValueError("review status must be approved, rejected, or superseded")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM harness_memories WHERE memory_id=?", (memory_id,)).fetchone()
            if row is None:
                raise ValueError(f"unknown memory: {memory_id}")
            conn.execute("UPDATE harness_memories SET status=?, content=?, updated_at=? WHERE memory_id=?",
                         (status, content if content is not None else row["content"], _now(), memory_id))
        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM harness_memories WHERE memory_id=?", (memory_id,))
        return cursor.rowcount > 0

    def export_memories(self, output_path: str | Path, approved_only: bool = False) -> int:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = self.list_memories(status="approved" if approved_only else None, limit=100000)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for item in items:
                stream.write(_dump({"memory_id": item.memory_id, "layer": item.layer, "content": item.content,
                                    "status": item.status, "source_task_id": item.source_task_id,
                                    "source_run_id": item.source_run_id, "confidence": item.confidence,
                                    "metadata": item.metadata, "created_at": item.created_at,
                                    "updated_at": item.updated_at}) + "\n")
        return len(items)

    def search_memories(self, query: str, limit: int = 20, include_pending: bool = False,
                        task_id: str | None = None) -> list[tuple[MemoryItem, float]]:
        terms = []
        for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", query):
            if "\u4e00" <= token[0] <= "\u9fff":
                terms.extend(token[index:index + 2] for index in range(max(1, len(token) - 1)))
            else:
                terms.append(token)
        terms = list(dict.fromkeys(term for term in terms if term))
        states = ("approved", "pending") if include_pending else ("approved",)
        state_marks = ",".join("?" for _ in states)
        task_filter = " AND (source_task_id=? OR source_task_id IS NULL)" if task_id else ""
        params: list[Any] = list(states)
        if task_id:
            params.append(task_id)
        rows = []
        with self._connect() as conn:
            if self.fts_available and terms:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                try:
                    rows = conn.execute(f"""SELECT m.*, bm25(harness_memory_fts) AS rank FROM harness_memory_fts f
                        JOIN harness_memories m ON m.memory_id=f.memory_id
                        WHERE f.content MATCH ? AND m.status IN ({state_marks}){task_filter}
                        ORDER BY rank LIMIT ?""", (expression, *params, limit)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                where = f"status IN ({state_marks}){task_filter}"
                if terms:
                    where += " AND (" + " OR ".join("content LIKE ?" for _ in terms) + ")"
                    params.extend(f"%{term}%" for term in terms)
                rows = conn.execute(f"SELECT * FROM harness_memories WHERE {where} ORDER BY updated_at DESC LIMIT ?",
                                    (*params, limit)).fetchall()
        return [(self._memory(row), 1.0 / (index + 1)) for index, row in enumerate(rows)]

    @staticmethod
    def _task(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(row["task_id"], row["title"], row["objective"], row["status"], json.loads(row["state_json"]), row["created_at"], row["updated_at"])

    @staticmethod
    def _memory(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(row["memory_id"], row["layer"], row["content"], row["status"], row["source_task_id"],
                          row["source_run_id"], float(row["confidence"]), json.loads(row["metadata_json"]),
                          None if row["embedding_json"] is None else json.loads(row["embedding_json"]), row["created_at"], row["updated_at"])

    def create_tool_proposal(self, proposal: dict[str, Any]) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute("INSERT INTO harness_tool_approvals VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?)",
                         (proposal["proposal_id"], proposal["task_id"], proposal["run_id"], proposal["server_name"],
                          proposal["tool_name"], _dump(proposal["arguments"]), int(proposal["side_effecting"]), now, now))

    def get_tool_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM harness_tool_approvals WHERE proposal_id=?", (proposal_id,)).fetchone()
        if row is None:
            return None
        return {"proposal_id": row["proposal_id"], "task_id": row["task_id"], "run_id": row["run_id"],
                "server_name": row["server_name"], "tool_name": row["tool_name"],
                "arguments": json.loads(row["arguments_json"]), "side_effecting": bool(row["side_effecting"]),
                "status": row["status"], "result": None if row["result_json"] is None else json.loads(row["result_json"])}

    def finish_tool_proposal(self, proposal_id: str, status: str, result: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            cursor = conn.execute("UPDATE harness_tool_approvals SET status=?, result_json=?, updated_at=? WHERE proposal_id=? AND status='pending'",
                                  (status, None if result is None else _dump(result), _now(), proposal_id))
            if not cursor.rowcount:
                raise ValueError("tool proposal is missing or no longer pending")
