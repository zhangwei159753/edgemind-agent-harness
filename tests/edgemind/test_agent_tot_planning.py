import pytest

from edgemind.harness.orchestration.planning import PlanCandidate, PlanStep, select_candidate, validate_candidates
from edgemind.harness.orchestration import PlanCandidate as PublicPlanCandidate


def candidate(candidate_id="plan-a", *, steps=None, goal=0.8, evidence=0.8, tools=0.8,
              risk=0.1, safe=True, cost=0.2):
    return {
        "candidate_id": candidate_id,
        "steps": steps or [{"step_id": "step-1", "goal": "Answer from evidence", "depends_on": [],
                             "expected_evidence": ["doc-1"], "expected_tools": ["web_search"],
                             "risk": "low", "estimated_calls": 1}],
        "goal_coverage": goal, "evidence_availability": evidence,
        "tool_availability": tools, "risk_score": risk, "safety_ok": safe,
        "estimated_cost": cost,
    }


def raw(*candidates):
    return {"candidates": list(candidates)}


def test_planning_contracts_are_exported_from_orchestration_package():
    assert PublicPlanCandidate is PlanCandidate


def test_valid_candidates_parse_to_immutable_structured_contracts():
    parsed = validate_candidates(raw(candidate()), {"web_search"})

    assert len(parsed) == 1
    assert isinstance(parsed[0], PlanCandidate)
    assert isinstance(parsed[0].steps[0], PlanStep)
    assert parsed[0].steps[0].expected_evidence == ("doc-1",)
    with pytest.raises((AttributeError, TypeError)):
        parsed[0].candidate_id = "changed"


@pytest.mark.parametrize("bad", [
    {"candidate_id": "plan-a", "extra": "scratchpad"},
    {"candidate_id": "plan-a", "steps": [], "goal_coverage": 1.1,
     "evidence_availability": 1, "tool_availability": 1, "risk_score": 0,
     "safety_ok": True, "estimated_cost": 0},
    {"candidate_id": "plan-a", "steps": [], "goal_coverage": 1,
     "evidence_availability": 1, "tool_availability": 1, "risk_score": 0,
     "safety_ok": True, "estimated_cost": 0, "raw_reasoning": "private"},
])
def test_invalid_candidate_schema_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_candidates(raw(bad), {"web_search"})


def test_candidate_limit_and_duplicate_candidate_ids_are_rejected():
    with pytest.raises(ValueError, match="candidate"):
        validate_candidates(raw(candidate("a"), candidate("b"), candidate("c"), candidate("d")),
                            {"web_search"})
    with pytest.raises(ValueError, match="duplicate"):
        validate_candidates(raw(candidate("a"), candidate("a")), {"web_search"})


def test_step_cycles_unknown_dependencies_and_forbidden_tools_are_rejected():
    cycle = [
        {"step_id": "a", "goal": "First", "depends_on": ["b"], "expected_evidence": [],
         "expected_tools": [], "risk": "low", "estimated_calls": 1},
        {"step_id": "b", "goal": "Second", "depends_on": ["a"], "expected_evidence": [],
         "expected_tools": [], "risk": "low", "estimated_calls": 1},
    ]
    with pytest.raises(ValueError, match="cycle"):
        validate_candidates(raw(candidate(steps=cycle)), set())
    missing_dependency = [{**cycle[0], "depends_on": ["missing"]}]
    with pytest.raises(ValueError, match="dependency"):
        validate_candidates(raw(candidate(steps=missing_dependency)), set())
    with pytest.raises(ValueError, match="tool"):
        validate_candidates(raw(candidate()), set())


def test_selector_applies_deterministic_score_and_simpler_plan_tiebreak():
    simple = candidate("b-simple")
    complex_steps = [
        {"step_id": "step-1", "goal": "Research", "depends_on": [], "expected_evidence": ["doc-1"],
         "expected_tools": ["web_search"], "risk": "low", "estimated_calls": 1},
        {"step_id": "step-2", "goal": "Synthesize", "depends_on": ["step-1"], "expected_evidence": ["doc-1"],
         "expected_tools": ["web_search"], "risk": "low", "estimated_calls": 1},
    ]
    complex_plan = candidate("a-complex", steps=complex_steps)
    parsed = validate_candidates(raw(complex_plan, simple), {"web_search"})

    selected, summary = select_candidate(parsed)

    assert selected.candidate_id == "b-simple"
    assert summary["selected_candidate_id"] == "b-simple"
    assert summary["score"] == pytest.approx(0.35 * 0.8 + 0.25 * 0.8 + 0.20 * 0.8 + 0.10 * 0.9 + 0.10 * 0.8)
    assert "raw_reasoning" not in str(summary)


def test_safety_is_a_hard_gate_and_empty_selection_has_reason():
    unsafe = validate_candidates(raw(candidate("unsafe", safe=False)), {"web_search"})[0]
    selected, summary = select_candidate((unsafe,))

    assert selected is None
    assert summary["reason"] == "no_safety_valid_candidate"
    assert summary["rejected_candidate_ids"] == ["unsafe"]
