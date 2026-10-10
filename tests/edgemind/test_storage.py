from edgemind.models import FailureAttribution, Feedback, QueryRun
from edgemind.storage import SqliteStore


def _run(run_id="run-1"):
    return QueryRun(
        run_id=run_id,
        created_at="2026-10-10T00:00:00Z",
        question="What was the revenue?",
        answer="Revenue was 10.",
        evidence=[{"chunk_id": "c1", "score": 0.9, "page": 3}],
        corpus_version="corpus-a",
        config_version="config-a",
        stage_ms={"retrieve": 12.5},
        status="succeeded",
        abstained=False,
        error_type=None,
    )


def test_store_round_trips_query_run_with_evidence_and_versions(tmp_path):
    store = SqliteStore(tmp_path / "state.sqlite3")
    store.record_run(_run())
    assert store.get_run("run-1") == _run()


def test_feedback_requires_existing_run_and_failed_insert_is_atomic(tmp_path):
    store = SqliteStore(tmp_path / "state.sqlite3")
    feedback = Feedback("fb-1", "missing", "wrong_citation", "", "2026-10-10T00:00:00Z")
    try:
        store.add_feedback(feedback)
    except ValueError:
        pass
    else:
        raise AssertionError("feedback for an unknown query must be rejected")
    assert store.list_feedback() == []
    store.record_run(_run())
    feedback = Feedback("fb-2", "run-1", "wrong_citation", "page mismatch", "2026-10-10T00:00:00Z")
    store.add_feedback(feedback)
    assert store.list_feedback() == [feedback]


def test_attributions_are_reviewable_and_require_existing_run(tmp_path):
    store = SqliteStore(tmp_path / "state.sqlite3")
    item = FailureAttribution("a-1", "missing", "empty_retrieval", "no chunks", "missing_document", False, None)
    try:
        store.record_attributions("missing", [item])
    except ValueError:
        pass
    else:
        raise AssertionError("attribution for an unknown query must be rejected")
    store.record_run(_run())
    item = FailureAttribution("a-1", "run-1", "empty_retrieval", "no chunks", "missing_document", False, None)
    store.record_attributions("run-1", [item])
    assert store.list_attributions("run-1") == [item]
    confirmed = store.confirm_attribution("a-1", "irrelevant_retrieval")
    assert confirmed.confirmed is True
    assert confirmed.reviewer_label == "irrelevant_retrieval"


def test_schema_initialization_is_idempotent(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    SqliteStore(db_path)
    store = SqliteStore(db_path)
    store.record_run(_run())
    store.record_experiment("exp-1", {"status": "candidate"})
    store.update_experiment("exp-1", {"status": "passed"})
    assert store.get_experiment("exp-1") == {"status": "passed"}
    store.record_version("v1", {"config": {"top_k_vector": 5}})
    store.set_active_version("v1")
    assert store.get_active_version() == {"version_id": "v1", "config": {"top_k_vector": 5}}
    store.record_corpus_manifest("corpus-a", {"chunk_ids": ["c1"]})
