import pytest

from edgemind.harness.policies import AgentPolicyService
from edgemind.harness.store import HarnessStore

REGISTERED = {
    "edgemind__get_weather", "edgemind__web_search", "ops__write_note",
}


def service_for(path):
    store = HarnessStore(path)
    return store, AgentPolicyService(store, registered_tool_ids=lambda: set(REGISTERED))


def seed_passed_experiment(store, candidate, experiment_id):
    store.save_agent_experiment({"experiment_id": experiment_id, "candidate_policy_id": candidate.policy_id,
                                 "baseline_policy_id": candidate.parent_policy_id or "baseline-v1",
                                 "dataset_hash": "test-dataset", "mode": "fixture", "passed": True,
                                 "created_at": "2026-10-10T00:00:00+00:00"})


def test_default_policy_candidate_versioning_promotion_and_rollback(tmp_path):
    store, policies = service_for(tmp_path / "state.sqlite3")
    baseline = policies.get_active()

    candidate = policies.create_candidate(
        {"memory_top_k": 4, "tool_allowlist": ["edgemind__get_weather"]},
        rationale="Reduce irrelevant memory and limit tools for a focused workflow.",
    )

    assert baseline.policy_id == "baseline-v1"
    assert candidate.version == 2
    assert candidate.parent_policy_id == baseline.policy_id
    assert candidate.memory_top_k == 4
    assert candidate.tool_allowlist == ("edgemind__get_weather",)
    assert policies.get(candidate.policy_id) == candidate
    with pytest.raises(ValueError, match="confirmation"):
        policies.promote(candidate.policy_id, "experiment-1", confirmed=False)
    assert policies.get_active() == baseline

    seed_passed_experiment(store, candidate, "experiment-1")
    promoted = policies.promote(candidate.policy_id, "experiment-1", confirmed=True)
    assert policies.get_active() == promoted
    restored = policies.rollback(baseline.policy_id, confirmed=True)
    assert restored == baseline
    assert policies.get_active() == baseline
    assert store.schema_version == 4
    assert [row["action"] for row in store.list_agent_policy_transitions()] == ["promote", "rollback"]


def test_policy_nested_configuration_is_immutable(tmp_path):
    _, policies = service_for(tmp_path / "state.sqlite3")
    policy = policies.get_active()

    with pytest.raises(TypeError):
        policy.layer_weights["semantic"] = 99
    with pytest.raises(TypeError):
        policy.output_constraints["max_output_chars"] = 1


@pytest.mark.parametrize("payload,rationale", [
    ({"unknown": "field"}, "rationale"),
    ({"python": "print(1)"}, "rationale"),
    ({"prompt_suffix": "api_key=sk-12345678901234567890"}, "rationale"),
    ({"prompt_suffix": "Authorization: Basic dXNlcjpwYXNzd29yZA=="}, "rationale"),
    ({"prompt_suffix": "safe"}, "token=secret-value"),
    ({"memory_top_k": 13}, "rationale"),
    ({"layer_weights": {"semantic": 3}}, "rationale"),
    ({"layer_weights": {"unknown": 0.5}}, "rationale"),
    ({"tool_allowlist": ["unknown__tool"]}, "rationale"),
    ({"max_tool_rounds": 4}, "rationale"),
    ({"time_decay_days": 0}, "rationale"),
    ({"output_constraints": {"run_shell": True}}, "rationale"),
    ({"checker_ids": ["custom_code"]}, "rationale"),
    ({"prompt_template_ref": "unregistered-template"}, "rationale"),
    ({"memory_filters": {"semantic": ["tag"]}}, "rationale"),
    ({"memory_top_k": 2}, "  "),
])
def test_invalid_candidate_is_rejected_without_changing_active_policy(tmp_path, payload, rationale):
    _, policies = service_for(tmp_path / "state.sqlite3")
    baseline = policies.get_active()

    with pytest.raises(ValueError):
        policies.create_candidate(payload, rationale)

    assert policies.get_active() == baseline
    assert len(policies.list_policies()) == 1


def test_candidate_may_not_expand_past_registered_tool_catalog(tmp_path):
    _, policies = service_for(tmp_path / "state.sqlite3")

    with pytest.raises(ValueError, match="registered"):
        policies.create_candidate(
            {"tool_allowlist": ["ops__write_note", "ops__delete_everything"]},
            "Only enable approved tools.",
        )


def test_baseline_includes_only_currently_registered_tools(tmp_path):
    _, policies = service_for(tmp_path / "state.sqlite3")

    policy = policies.get_active()

    assert set(policy.tool_allowlist) == REGISTERED
