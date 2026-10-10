import pytest

from edgemind.experiments import ExperimentService, QualityGates
from edgemind.models import RetrievalConfig
from edgemind.storage import SqliteStore


class FakeEvaluation:
    def __init__(self, metrics):
        self.metrics = metrics
        self.calls = []

    def run(self, dataset_path, config=None, config_version="baseline"):
        self.calls.append((dataset_path, config, config_version))
        return {"metrics": self.metrics, "sample_count": 2}


def _setup(tmp_path, candidate_metrics):
    store = SqliteStore(tmp_path / "state.sqlite3")
    store.record_version("v1", {
        "config": {"top_k_vector": 10},
        "metrics": {"recall_at_k": 0.8, "citation_hit_rate": 0.9},
    })
    store.set_active_version("v1")
    evaluator = FakeEvaluation(candidate_metrics)
    gates = QualityGates(required_min={"recall_at_k": 0.7}, max_regression={"citation_hit_rate": 0.05})
    return store, evaluator, ExperimentService(store, evaluator, gates)


def test_candidate_is_recorded_with_config_and_metric_diff(tmp_path):
    store, evaluator, service = _setup(tmp_path, {"recall_at_k": 0.9, "citation_hit_rate": 0.92})
    result = service.evaluate_candidate(RetrievalConfig(top_k_vector=20), "dataset.jsonl")
    assert result["status"] == "passed"
    assert result["metric_diff"]["recall_at_k"] == pytest.approx(0.1)
    assert store.get_experiment(result["experiment_id"])["config"]["top_k_vector"] == 20
    assert evaluator.calls[0][1].top_k_vector == 20
    assert evaluator.calls[0][2] == "candidate"


def test_missing_required_metric_blocks_promotion(tmp_path):
    store, _, service = _setup(tmp_path, {"recall_at_k": None, "citation_hit_rate": 0.9})
    result = service.evaluate_candidate(RetrievalConfig(), "dataset.jsonl")
    assert result["status"] == "blocked"
    with pytest.raises(ValueError, match="not promotable"):
        service.promote(result["experiment_id"])
    assert store.get_active_version()["version_id"] == "v1"


def test_regression_blocks_promotion_and_active_version_is_unchanged(tmp_path):
    store, _, service = _setup(tmp_path, {"recall_at_k": 0.75, "citation_hit_rate": 0.8})
    result = service.evaluate_candidate(RetrievalConfig(), "dataset.jsonl")
    assert result["status"] == "blocked"
    assert any("citation_hit_rate" in reason for reason in result["gate_reasons"])
    assert store.get_active_version()["version_id"] == "v1"


def test_promotion_and_rollback_create_auditable_versions(tmp_path):
    store, _, service = _setup(tmp_path, {"recall_at_k": 0.9, "citation_hit_rate": 0.95})
    result = service.evaluate_candidate(RetrievalConfig(), "dataset.jsonl")
    promoted = service.promote(result["experiment_id"])
    assert store.get_active_version()["version_id"] == promoted
    assert store.get_experiment(result["experiment_id"])["status"] == "promoted"
    assert service.rollback("v1") == "v1"
    assert store.get_active_version()["version_id"] == "v1"
    rolled_back = store.get_experiment(result["experiment_id"])
    assert rolled_back["status"] == "rolled_back"
    assert rolled_back["rollback_to"] == "v1"
