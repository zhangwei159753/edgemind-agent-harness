import json

from edgemind.evaluation import EvaluationRunner, mean_reciprocal_rank, recall_at_k
from edgemind.models import QueryRun


def test_recall_at_k_and_mrr_for_ranked_relevant_chunks():
    assert recall_at_k(["a", "b"], ["b", "c"], k=2) == 0.5
    assert mean_reciprocal_rank(["a", "b"], ["b"]) == 0.5


def _dataset(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class FakeQueryService:
    def ask(self, question, top_k=5):
        if question == "boom":
            raise TimeoutError("provider unavailable")
        values = {
            "q1": ("answer one", ["c1"], ["c1"], False),
            "q2": ("", [], [], False),
            "q3": ("answer three", ["c3"], [], None),
        }
        answer, retrieved_ids, cited_ids, abstained = values[question]
        return QueryRun(
            run_id=question, created_at="2026-10-10T00:00:00Z", question=question, answer=answer,
            evidence=[{"chunk_id": value} for value in retrieved_ids], citations=[{"chunk_id": value} for value in cited_ids],
            corpus_version="c1", config_version="v1", stage_ms={"total": 1.0}, status="succeeded", abstained=abstained,
        )


def test_unlabeled_retrieval_metrics_are_null(tmp_path):
    dataset = _dataset(tmp_path / "unlabeled.jsonl", [{"example_id": "e1", "question": "q1"}])
    result = EvaluationRunner(lambda config, version: FakeQueryService()).run(dataset)
    assert result["metrics"]["recall_at_k"] is None
    assert result["metrics"]["mrr"] is None
    assert result["metrics"]["citation_hit_rate"] is None
    assert result["metrics"]["no_answer_accuracy"] is None


def test_no_answer_accuracy_counts_only_labeled_samples(tmp_path):
    rows = [
        {"example_id": "e1", "question": "q1", "expected_chunk_ids": ["c1"], "answerable": True},
        {"example_id": "e2", "question": "q2", "expected_chunk_ids": [], "answerable": False},
        {"example_id": "e3", "question": "q3", "expected_chunk_ids": ["c3"]},
    ]
    result = EvaluationRunner(lambda config, version: FakeQueryService()).run(_dataset(tmp_path / "labels.jsonl", rows))
    assert result["metrics"]["no_answer_accuracy"] == 0.5


def test_provider_failure_is_recorded_and_later_sample_still_runs(tmp_path):
    rows = [
        {"example_id": "e1", "question": "boom", "expected_chunk_ids": ["c1"]},
        {"example_id": "e2", "question": "q1", "expected_chunk_ids": ["c1"]},
    ]
    result = EvaluationRunner(lambda config, version: FakeQueryService()).run(_dataset(tmp_path / "failure.jsonl", rows))
    assert len(result["samples"]) == 2
    assert result["samples"][0]["error_type"] == "TimeoutError"
    assert result["samples"][1]["retrieved_ids"] == ["c1"]
    assert result["failure_count"] == 1
