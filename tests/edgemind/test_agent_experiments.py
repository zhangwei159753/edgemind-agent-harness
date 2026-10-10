import pytest

from edgemind.agent_experiments import AgentExperimentService, AgentQualityGates
from edgemind.harness.policies import AgentPolicyService
from edgemind.harness.store import HarnessStore


class FakeEvaluationRunner:
    def __init__(self, by_policy=None, error=None):
        self.by_policy = by_policy or {}
        self.error = error
        self.calls = []
    def run(self, dataset_path, policy, mode="fixture", repetitions=1, confirmed_live=False):
        self.calls.append((str(dataset_path), policy.policy_id, mode, repetitions))
        if self.error:
            raise RuntimeError(self.error)
        return self.by_policy[policy.policy_id]


def make_service(tmp_path, candidate_metrics, baseline_metrics=None, gates=None, error=None):
    store = HarnessStore(tmp_path / "state.sqlite3")
    policies = AgentPolicyService(store)
    baseline = policies.get_active()
    candidate = policies.create_candidate({"prompt_suffix": "Candidate policy."}, "Compare a measured candidate.")
    common = {"dataset_hash": "same-dataset", "mode": "fixture", "case_results": []}
    reports = {
        baseline.policy_id: {**common, "policy_id": baseline.policy_id, "metrics": baseline_metrics or {
            "case_pass_rate": 0.8, "tool_call_success_rate": 1.0, "latency_ms_mean": 500.0,
            "forbidden_tool_attempts": 0}},
        candidate.policy_id: {**common, "policy_id": candidate.policy_id, "metrics": candidate_metrics},
    }
    runner = FakeEvaluationRunner(reports, error=error)
    service = AgentExperimentService(policies, runner, store,
                                     gates or AgentQualityGates(required_min={"case_pass_rate": 0.85},
                                                                max_regression={"tool_call_success_rate": 0.05},
                                                                max_latency_ms=1000))
    return store, policies, candidate, runner, service


def test_same_dataset_candidate_gates_must_pass_before_confirmed_promotion(tmp_path):
    store, policies, candidate, runner, experiments = make_service(tmp_path, {
        "case_pass_rate": 0.9, "tool_call_success_rate": 0.95, "latency_ms_mean": 700.0,
        "forbidden_tool_attempts": 0,
    })

    report = experiments.evaluate_candidate(candidate.policy_id, "dataset.jsonl")

    assert report["passed"] is True
    assert report["baseline"]["dataset_hash"] == report["candidate"]["dataset_hash"]
    assert [call[1] for call in runner.calls] == ["baseline-v1", candidate.policy_id]
    assert store.schema_version == 4
    with pytest.raises(ValueError, match="confirmation"):
        experiments.promote(report["experiment_id"], confirmed=False)
    assert policies.get_active().policy_id == "baseline-v1"
    promoted = experiments.promote(report["experiment_id"], confirmed=True)
    assert promoted.policy_id == candidate.policy_id
    assert policies.get_active().policy_id == candidate.policy_id


def test_missing_metrics_regression_and_unsafe_attempts_block_promotion(tmp_path):
    store, policies, candidate, _, experiments = make_service(tmp_path, {
        "case_pass_rate": None, "tool_call_success_rate": 0.7, "latency_ms_mean": 1500.0,
        "forbidden_tool_attempts": 1,
    })

    report = experiments.evaluate_candidate(candidate.policy_id, "dataset.jsonl")

    assert report["passed"] is False
    assert any("case_pass_rate" in reason for reason in report["gate_reasons"])
    assert any("forbidden" in reason.lower() for reason in report["gate_reasons"])
    with pytest.raises(ValueError, match="gate"):
        experiments.promote(report["experiment_id"], confirmed=True)
    with pytest.raises(ValueError, match="gate"):
        policies.promote(candidate.policy_id, report["experiment_id"], confirmed=True)
    assert policies.get_active().policy_id == "baseline-v1"
    assert store.get_agent_experiment(report["experiment_id"])["passed"] is False


def test_evaluator_failure_is_persisted_and_does_not_change_active_policy(tmp_path):
    store, policies, candidate, _, experiments = make_service(
        tmp_path, {}, error="fixture evaluator crashed"
    )

    report = experiments.evaluate_candidate(candidate.policy_id, "dataset.jsonl")

    assert report["passed"] is False
    assert report["error_category"] == "RuntimeError"
    assert store.get_agent_experiment(report["experiment_id"])["passed"] is False
    assert policies.get_active().policy_id == "baseline-v1"


def test_rollback_requires_confirmation_and_restores_existing_policy(tmp_path):
    store, policies, candidate, _, experiments = make_service(tmp_path, {
        "case_pass_rate": 0.95, "tool_call_success_rate": 1.0, "latency_ms_mean": 400.0,
        "forbidden_tool_attempts": 0,
    })
    report = experiments.evaluate_candidate(candidate.policy_id, "dataset.jsonl")
    experiments.promote(report["experiment_id"], confirmed=True)

    with pytest.raises(ValueError, match="confirmation"):
        experiments.rollback("baseline-v1", confirmed=False)
    restored = experiments.rollback("baseline-v1", confirmed=True)

    assert restored.policy_id == "baseline-v1"
    assert policies.get_active().policy_id == "baseline-v1"
    assert store.list_agent_policy_transitions()[-1]["action"] == "rollback"


def test_rollback_rejects_candidate_that_was_never_active(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    policies = AgentPolicyService(store)
    candidate = policies.create_candidate({"prompt_suffix": "Untested."}, "Never promoted")
    experiments = AgentExperimentService(policies, None, store, AgentQualityGates())

    with pytest.raises(ValueError, match="previously active"):
        experiments.rollback(candidate.policy_id, confirmed=True)

    assert policies.get_active().policy_id == "baseline-v1"


def test_evaluator_errors_and_unexpected_failed_cases_block_candidate(tmp_path):
    store, policies, candidate, runner, experiments = make_service(tmp_path, {
        "case_pass_rate": 6 / 7, "tool_call_success_rate": 1.0, "latency_ms_mean": 700.0,
        "forbidden_tool_attempts": 0,
    })
    runner.by_policy[candidate.policy_id]["case_results"] = [{
        "example_id": "mcp_confirmation", "passed": False, "actual_status": "failed",
        "expected_outcome": "awaiting_tool_approval", "unsafe_tool_attempts": 0,
        "error_category": "IndexError",
    }]

    report = experiments.evaluate_candidate(candidate.policy_id, "dataset.jsonl")

    assert report["passed"] is False
    assert any("failed case" in reason.lower() or "evaluator error" in reason.lower()
               for reason in report["gate_reasons"])


def test_quality_gate_configuration_rejects_non_finite_thresholds():
    with pytest.raises(ValueError):
        AgentQualityGates(required_min={"case_pass_rate": float("nan")})
    with pytest.raises(ValueError):
        AgentQualityGates(max_regression={"latency": float("inf")})
