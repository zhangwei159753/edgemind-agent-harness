import json
from dataclasses import asdict

import pytest

from edgemind.feedback import FeedbackService
from edgemind.models import QueryRun
from edgemind.storage import SqliteStore


def _store_with_run(path):
    store = SqliteStore(path)
    store.record_run(QueryRun("r1", "2026-10-10T00:00:00Z", "q", "a", [], "c1", "v1", {}, "succeeded"))
    return store


def test_feedback_rejects_unknown_label_and_round_trips_jsonl(tmp_path):
    service = FeedbackService(_store_with_run(tmp_path / "state.sqlite3"))
    with pytest.raises(ValueError, match="unknown feedback label"):
        service.submit("r1", "bad_label")
    service.submit("r1", "wrong_citation", "page mismatch", feedback_id="fb1")
    output = tmp_path / "feedback.jsonl"
    assert service.export_jsonl(output) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == asdict(service.store.list_feedback()[0])
    other = FeedbackService(_store_with_run(tmp_path / "other.sqlite3"))
    assert other.import_jsonl(output) == 1
    assert other.store.list_feedback()[0].feedback_id == "fb1"


def test_import_jsonl_reports_line_error_without_silently_dropping_record(tmp_path):
    service = FeedbackService(_store_with_run(tmp_path / "state.sqlite3"))
    path = tmp_path / "bad.jsonl"
    valid = '{"feedback_id":"fb1","run_id":"r1","label":"wrong_citation","note":"","created_at":"2026-10-10T00:00:00Z"}'
    path.write_text(valid + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        service.import_jsonl(path)
    assert service.store.list_feedback() == []
