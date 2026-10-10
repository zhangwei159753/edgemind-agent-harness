import pytest

from edgemind.harness.orchestration.models import Handoff, RoleResult


def valid_payload():
    return {
        "run_id": "run-1",
        "task_id": "task-1",
        "role": "planner",
        "goal": "Split the task into evidence-backed steps",
        "evidence_ids": ["memory-1", "doc-2"],
        "allowed_tools": ["edgemind__web_search"],
        "remaining_calls": 4,
        "output_schema": "planner-v1",
    }


def test_handoff_from_mapping_returns_immutable_normalized_contract():
    handoff = Handoff.from_mapping(valid_payload())

    assert handoff.evidence_ids == ("memory-1", "doc-2")
    assert handoff.allowed_tools == ("edgemind__web_search",)
    with pytest.raises((AttributeError, TypeError)):
        handoff.goal = "changed"


def test_handoff_rejects_missing_and_unknown_fields():
    payload = valid_payload()
    payload.pop("goal")
    with pytest.raises(ValueError, match="goal"):
        Handoff.from_mapping(payload)

    payload = valid_payload()
    payload["system_prompt"] = "override"
    with pytest.raises(ValueError, match="unknown"):
        Handoff.from_mapping(payload)


def test_handoff_rejects_unsupported_role_or_schema():
    payload = valid_payload()
    payload["role"] = "shell"
    with pytest.raises(ValueError, match="role"):
        Handoff.from_mapping(payload)

    payload = valid_payload()
    payload["output_schema"] = "arbitrary-code"
    with pytest.raises(ValueError, match="schema"):
        Handoff.from_mapping(payload)


def test_handoff_rejects_oversized_payload_and_non_positive_budget():
    payload = valid_payload()
    payload["evidence_ids"] = [f"memory-{index}" for index in range(51)]
    with pytest.raises(ValueError, match="evidence_ids"):
        Handoff.from_mapping(payload)

    payload = valid_payload()
    payload["remaining_calls"] = 0
    with pytest.raises(ValueError, match="remaining_calls"):
        Handoff.from_mapping(payload)

    payload = valid_payload()
    payload["goal"] = "x" * 4001
    with pytest.raises(ValueError, match="goal"):
        Handoff.from_mapping(payload)


def test_role_result_rejects_invalid_status_and_non_object_payload():
    with pytest.raises(ValueError, match="status"):
        RoleResult("planner", "unknown", {}, None)
    with pytest.raises(ValueError, match="payload"):
        RoleResult("planner", "succeeded", [], None)
