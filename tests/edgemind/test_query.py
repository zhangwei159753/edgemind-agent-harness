from edgemind.query import QueryService
from edgemind.storage import SqliteStore


class FakeRag:
    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def generate_answer(self, question, top_k=5):
        if self.error:
            raise self.error
        return self.result


def test_query_service_persists_evidence_and_stage_timings_with_fake_rag(tmp_path):
    result = {
        "answer": "Revenue was 10.",
        "retrieval_chunks": [{"chunk_id": "c1", "content": "Revenue: 10", "score": 0.9}],
        "citation_ids": ["c1"],
        "stage_ms": {"retrieval": 4.0, "generation": 8.0},
        "abstained": False,
    }
    store = SqliteStore(tmp_path / "state.sqlite3")
    service = QueryService(lambda config: FakeRag(result=result), store, "corpus-a", "config-a")
    run = service.ask("What was revenue?")
    assert run.status == "succeeded"
    assert run.evidence == result["retrieval_chunks"]
    assert run.stage_ms["retrieval"] == 4.0
    assert store.get_run(run.run_id) == run


def test_query_service_records_provider_failure_as_failed_run(tmp_path):
    store = SqliteStore(tmp_path / "state.sqlite3")
    service = QueryService(lambda config: FakeRag(error=TimeoutError()), store, "corpus-a", "config-a")
    run = service.ask("What was revenue?")
    assert run.status == "failed"
    assert run.answer == ""
    assert run.error_type == "TimeoutError"
    assert store.get_run(run.run_id) == run


def test_rule_diagnosis_flags_empty_retrieval_as_reviewable_suggestion(tmp_path):
    store = SqliteStore(tmp_path / "state.sqlite3")
    service = QueryService(lambda config: FakeRag(result={"answer": "", "retrieval_chunks": []}),
                           store, "corpus-a", "config-a")
    run = service.ask("What was revenue?")
    items = store.list_attributions(run.run_id)
    assert len(items) == 1
    assert items[0].category == "empty_retrieval"
    assert items[0].suggested_label == "missing_document"
    assert items[0].confirmed is False


def test_legacy_retrieved_chunks_receive_stable_ids_and_citations_map_to_them(tmp_path):
    store = SqliteStore(tmp_path / "state.sqlite3")
    result = {
        "answer": "Revenue was 10.",
        "retrieval_chunks": [{"content": "Revenue: 10", "metadata": {"file_name": "annual.pdf", "page": 3}}],
        "citations": [{"filename": "annual.pdf", "page": 3}],
    }
    service = QueryService(lambda config: FakeRag(result=result), store, "corpus-a", "config-a")
    run = service.ask("What was revenue?")
    assert run.evidence[0]["chunk_id"].startswith("chk_")
    assert run.citations[0]["chunk_id"] == run.evidence[0]["chunk_id"]
