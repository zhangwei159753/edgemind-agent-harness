from pathlib import Path

from edgemind.agent_eval_fixtures import build_evaluation_runner
from edgemind.agent_experiments import AgentExperimentService, AgentQualityGates
from edgemind.harness.memory import MemoryService
from edgemind.harness.policies import AgentPolicyService
from edgemind.harness.quality import QualityService
from edgemind.harness.store import HarnessStore
from edgemind.harness.tools import ToolRegistry


DATASET = Path("tests/data/agent_multi_agent_v1.jsonl")


def test_same_dataset_single_multi_report_and_quality_gate(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    policies = AgentPolicyService(store)
    services = {"store": store, "memory": MemoryService(store), "tools": ToolRegistry(store),
                "quality": QualityService(store)}
    runner = build_evaluation_runner(services)
    experiments = AgentExperimentService(policies, runner, store, AgentQualityGates())

    report = experiments.evaluate_orchestration_modes("baseline-v1", DATASET, max_latency_ms=100_000)

    assert report["single"]["dataset_hash"] == report["multi"]["dataset_hash"] == report["dataset_hash"]
    assert report["single"]["orchestration_mode"] == "single"
    assert report["multi"]["orchestration_mode"] == "multi"
    assert report["single"]["metrics"]["case_pass_rate"] == 1.0
    assert report["multi"]["metrics"]["case_pass_rate"] == 1.0
    assert report["single"]["metrics"]["model_call_count_mean"] == 1
    assert report["multi"]["metrics"]["model_call_count_mean"] == 13 / 6
    assert report["multi"]["metrics"]["latency_ms_p95"] is not None
    assert report["multi"]["metrics"]["token_usage_mean"] is None
    assert report["multi"]["na_reasons"]["token_usage_mean"]
    assert report["case_differences"] == []
    assert report["quality_gate"]["passed"] is True
    cases = {case["example_id"]: case for case in report["multi"]["case_results"]}
    assert cases["simple_single_path"]["actual_status"] == "succeeded"
    assert cases["reviewer_evidence_gap"]["role_sequence"] == ["planner", "researcher", "executor", "reviewer"]
    assert cases["unsafe_plan_rejected"]["actual_status"] == "needs_clarification"
    assert cases["unsafe_plan_rejected"]["selected_plan_id"] is None
    assert cases["planner_budget_exhaustion"]["model_call_count"] == 1
    assert cases["planner_provider_failure"]["model_call_count"] == 2


def test_comparison_gate_blocks_p95_latency_over_budget(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    policies = AgentPolicyService(store)
    services = {"store": store, "memory": MemoryService(store), "tools": ToolRegistry(store),
                "quality": QualityService(store)}
    runner = build_evaluation_runner(services)
    experiments = AgentExperimentService(policies, runner, store, AgentQualityGates())

    report = experiments.evaluate_orchestration_modes("baseline-v1", DATASET, max_latency_ms=0.000001)

    assert report["quality_gate"]["passed"] is False
    assert any("p95 latency" in reason for reason in report["quality_gate"]["reasons"])
