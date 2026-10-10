import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import FailureAttribution, Feedback, QueryRun


FEEDBACK_LABELS = {"correct", "missing_evidence", "wrong_citation", "irrelevant_retrieval", "missing_document", "ungrounded_answer"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SqliteStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS query_runs (
                    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, question TEXT NOT NULL,
                    answer TEXT NOT NULL, evidence_json TEXT NOT NULL, corpus_version TEXT NOT NULL,
                    config_version TEXT NOT NULL, stage_ms_json TEXT NOT NULL, status TEXT NOT NULL,
                    abstained INTEGER, error_type TEXT, citations_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES query_runs(run_id),
                    label TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attributions (
                    attribution_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES query_runs(run_id),
                    category TEXT NOT NULL, reason TEXT NOT NULL, suggested_label TEXT,
                    confirmed INTEGER NOT NULL DEFAULT 0, reviewer_label TEXT
                );
                CREATE TABLE IF NOT EXISTS experiments (experiment_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS config_versions (
                    version_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS corpus_manifests (
                    corpus_version TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(query_runs)")}
            if "citations_json" not in columns:
                conn.execute("ALTER TABLE query_runs ADD COLUMN citations_json TEXT NOT NULL DEFAULT '[]'")

    def record_run(self, run: QueryRun) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO query_runs
                (run_id, created_at, question, answer, evidence_json, corpus_version,
                 config_version, stage_ms_json, status, abstained, error_type, citations_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run.run_id, run.created_at, run.question, run.answer, _json(run.evidence), run.corpus_version,
                 run.config_version, _json(run.stage_ms), run.status,
                 None if run.abstained is None else int(run.abstained), run.error_type, _json(run.citations)),
            )

    def get_run(self, run_id: str) -> QueryRun | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM query_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return QueryRun(
            run_id=row["run_id"], created_at=row["created_at"], question=row["question"], answer=row["answer"],
            evidence=json.loads(row["evidence_json"]), corpus_version=row["corpus_version"],
            config_version=row["config_version"], stage_ms=json.loads(row["stage_ms_json"]), status=row["status"],
            abstained=None if row["abstained"] is None else bool(row["abstained"]), error_type=row["error_type"],
            citations=json.loads(row["citations_json"]),
        )

    def add_feedback(self, feedback: Feedback) -> None:
        if feedback.label not in FEEDBACK_LABELS:
            raise ValueError(f"unknown feedback label: {feedback.label}")
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM query_runs WHERE run_id = ?", (feedback.run_id,)).fetchone()
            if exists is None:
                raise ValueError(f"unknown query run: {feedback.run_id}")
            conn.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?)",
                         (feedback.feedback_id, feedback.run_id, feedback.label, feedback.note, feedback.created_at))

    def list_feedback(self, label: str | None = None) -> list[Feedback]:
        with self._connect() as conn:
            if label is None:
                rows = conn.execute("SELECT * FROM feedback ORDER BY created_at, feedback_id").fetchall()
            else:
                rows = conn.execute("SELECT * FROM feedback WHERE label = ? ORDER BY created_at, feedback_id", (label,)).fetchall()
        return [Feedback(row["feedback_id"], row["run_id"], row["label"], row["note"], row["created_at"]) for row in rows]

    def record_attributions(self, run_id: str, items: list[FailureAttribution]) -> None:
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM query_runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise ValueError(f"unknown query run: {run_id}")
            for item in items:
                if item.run_id != run_id:
                    raise ValueError("attribution run_id does not match the requested run")
                conn.execute("INSERT INTO attributions VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (item.attribution_id, run_id, item.category, item.reason, item.suggested_label,
                              int(item.confirmed), item.reviewer_label))

    def list_attributions(self, run_id: str) -> list[FailureAttribution]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM attributions WHERE run_id = ? ORDER BY rowid", (run_id,)).fetchall()
        return [FailureAttribution(row["attribution_id"], row["run_id"], row["category"], row["reason"],
                                   row["suggested_label"], bool(row["confirmed"]), row["reviewer_label"]) for row in rows]

    def confirm_attribution(self, attribution_id: str, reviewer_label: str) -> FailureAttribution:
        if reviewer_label not in FEEDBACK_LABELS:
            raise ValueError(f"unknown feedback label: {reviewer_label}")
        with self._connect() as conn:
            cursor = conn.execute("UPDATE attributions SET confirmed = 1, reviewer_label = ? WHERE attribution_id = ?",
                                  (reviewer_label, attribution_id))
            if cursor.rowcount == 0:
                raise ValueError(f"unknown attribution: {attribution_id}")
            row = conn.execute("SELECT * FROM attributions WHERE attribution_id = ?", (attribution_id,)).fetchone()
        return FailureAttribution(row["attribution_id"], row["run_id"], row["category"], row["reason"],
                                  row["suggested_label"], bool(row["confirmed"]), row["reviewer_label"])

    def record_experiment(self, experiment_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO experiments VALUES (?, ?)", (experiment_id, _json(payload)))

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def update_experiment(self, experiment_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            cursor = conn.execute("UPDATE experiments SET payload_json = ? WHERE experiment_id = ?",
                                  (_json(payload), experiment_id))
            if cursor.rowcount == 0:
                raise ValueError(f"unknown experiment: {experiment_id}")

    def record_version(self, version_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO config_versions (version_id, payload_json) VALUES (?, ?)",
                         (version_id, _json(payload)))

    def get_active_version(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT version_id, payload_json FROM config_versions WHERE is_active = 1").fetchone()
        return None if row is None else {"version_id": row["version_id"], **json.loads(row["payload_json"])}

    def set_active_version(self, version_id: str) -> None:
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM config_versions WHERE version_id = ?", (version_id,)).fetchone() is None:
                raise ValueError(f"unknown config version: {version_id}")
            conn.execute("UPDATE config_versions SET is_active = 0 WHERE is_active = 1")
            conn.execute("UPDATE config_versions SET is_active = 1 WHERE version_id = ?", (version_id,))

    def record_corpus_manifest(self, corpus_version: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO corpus_manifests VALUES (?, ?, CURRENT_TIMESTAMP)",
                         (corpus_version, _json(payload)))
