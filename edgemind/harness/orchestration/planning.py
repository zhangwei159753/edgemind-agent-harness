from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


_STEP_FIELDS = {"step_id", "goal", "depends_on", "expected_evidence", "expected_tools", "risk", "estimated_calls"}
_CANDIDATE_FIELDS = {
    "candidate_id", "steps", "goal_coverage", "evidence_availability", "tool_availability",
    "risk_score", "safety_ok", "estimated_cost",
}
_RELATIVE_SCORE_WEIGHTS = {
    "goal_coverage": 0.35,
    "evidence_availability": 0.25,
    "tool_availability": 0.20,
    "risk_score": -0.10,
    "estimated_cost": -0.10,
}


def _text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be non-empty text no longer than {maximum} characters")
    return value


def _string_tuple(value: Any, name: str, maximum: int, item_length: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name} must be a list with at most {maximum} items")
    values = tuple(_text(item, name, item_length) for item in value)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _normalized(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite number from 0 to 1")
    return float(value)


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    goal: str
    depends_on: tuple[str, ...]
    expected_evidence: tuple[str, ...]
    expected_tools: tuple[str, ...]
    risk: str
    estimated_calls: int

    def __post_init__(self) -> None:
        _text(self.step_id, "step_id", 80)
        _text(self.goal, "step goal", 1000)
        for name, value, maximum, length in (
            ("depends_on", self.depends_on, 5, 80),
            ("expected_evidence", self.expected_evidence, 20, 200),
            ("expected_tools", self.expected_tools, 20, 120),
        ):
            if not isinstance(value, tuple) or len(value) > maximum:
                raise ValueError(f"{name} must be a tuple with at most {maximum} items")
            if any(not isinstance(item, str) or not item.strip() or len(item) > length for item in value):
                raise ValueError(f"{name} contains an invalid item")
            if len(value) != len(set(value)):
                raise ValueError(f"{name} must not contain duplicates")
        if self.risk not in {"low", "medium", "high"}:
            raise ValueError("risk must be low, medium, or high")
        if (not isinstance(self.estimated_calls, int) or isinstance(self.estimated_calls, bool)
                or not 1 <= self.estimated_calls <= 8):
            raise ValueError("estimated_calls must be an integer from 1 to 8")


@dataclass(frozen=True)
class PlanCandidate:
    candidate_id: str
    steps: tuple[PlanStep, ...]
    goal_coverage: float
    evidence_availability: float
    tool_availability: float
    risk_score: float
    safety_ok: bool
    estimated_cost: float

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id", 80)
        if not isinstance(self.steps, tuple) or not 1 <= len(self.steps) <= 5:
            raise ValueError("candidate must contain 1 to 5 steps")
        if any(not isinstance(step, PlanStep) for step in self.steps):
            raise ValueError("candidate steps must be PlanStep values")
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise ValueError("candidate step IDs must be unique")
        for name in ("goal_coverage", "evidence_availability", "tool_availability", "risk_score", "estimated_cost"):
            object.__setattr__(self, name, _normalized(getattr(self, name), name))
        if not isinstance(self.safety_ok, bool):
            raise ValueError("safety_ok must be a boolean")


def _validate_step_graph(steps: tuple[PlanStep, ...]) -> None:
    by_id = {step.step_id: step for step in steps}
    for step in steps:
        unknown = set(step.depends_on) - set(by_id)
        if unknown:
            raise ValueError("step contains an undeclared dependency")
        if step.step_id in step.depends_on:
            raise ValueError("step dependency cycle detected")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("step dependency cycle detected")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in by_id[step_id].depends_on:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in by_id:
        visit(step_id)


def validate_candidates(raw: object, allowed_tools: set[str], max_steps: int = 5,
                        max_candidates: int = 3) -> tuple[PlanCandidate, ...]:
    if not isinstance(raw, dict) or set(raw) != {"candidates"}:
        raise ValueError("plan output must contain only a candidates list")
    if not isinstance(allowed_tools, set) or any(not isinstance(item, str) for item in allowed_tools):
        raise ValueError("allowed_tools must be a set of names")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 5:
        raise ValueError("max_steps must be an integer from 1 to 5")
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or not 1 <= max_candidates <= 3:
        raise ValueError("max_candidates must be an integer from 1 to 3")
    raw_candidates = raw["candidates"]
    if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= max_candidates:
        raise ValueError(f"candidate count must be from 1 to {max_candidates}")

    candidates: list[PlanCandidate] = []
    seen_candidate_ids: set[str] = set()
    for candidate_payload in raw_candidates:
        if not isinstance(candidate_payload, dict) or set(candidate_payload) != _CANDIDATE_FIELDS:
            raise ValueError("candidate contains missing or unknown fields")
        candidate_id = _text(candidate_payload["candidate_id"], "candidate_id", 80)
        if candidate_id in seen_candidate_ids:
            raise ValueError("duplicate candidate ID")
        seen_candidate_ids.add(candidate_id)
        raw_steps = candidate_payload["steps"]
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= max_steps:
            raise ValueError(f"candidate steps must be from 1 to {max_steps}")
        steps = []
        for step_payload in raw_steps:
            if not isinstance(step_payload, dict) or set(step_payload) != _STEP_FIELDS:
                raise ValueError("step contains missing or unknown fields")
            step = PlanStep(
                _text(step_payload["step_id"], "step_id", 80),
                _text(step_payload["goal"], "step goal", 1000),
                _string_tuple(step_payload["depends_on"], "depends_on", 5, 80),
                _string_tuple(step_payload["expected_evidence"], "expected_evidence", 20, 200),
                _string_tuple(step_payload["expected_tools"], "expected_tools", 20, 120),
                step_payload["risk"], step_payload["estimated_calls"],
            )
            if not set(step.expected_tools).issubset(allowed_tools):
                raise ValueError("candidate contains a forbidden tool")
            steps.append(step)
        step_tuple = tuple(steps)
        _validate_step_graph(step_tuple)
        candidates.append(PlanCandidate(
            candidate_id, step_tuple,
            _normalized(candidate_payload["goal_coverage"], "goal_coverage"),
            _normalized(candidate_payload["evidence_availability"], "evidence_availability"),
            _normalized(candidate_payload["tool_availability"], "tool_availability"),
            _normalized(candidate_payload["risk_score"], "risk_score"),
            candidate_payload["safety_ok"],
            _normalized(candidate_payload["estimated_cost"], "estimated_cost"),
        ))
    return tuple(candidates)


def candidate_score(candidate: PlanCandidate) -> float:
    return (
        0.35 * candidate.goal_coverage
        + 0.25 * candidate.evidence_availability
        + 0.20 * candidate.tool_availability
        + 0.10 * (1 - candidate.risk_score)
        + 0.10 * (1 - candidate.estimated_cost)
    )


def select_candidate(candidates: tuple[PlanCandidate, ...]) -> tuple[PlanCandidate | None, dict]:
    if not isinstance(candidates, tuple) or any(not isinstance(item, PlanCandidate) for item in candidates):
        raise ValueError("candidates must be a tuple of PlanCandidate values")
    safe = [candidate for candidate in candidates if candidate.safety_ok]
    if not safe:
        return None, {"selected_candidate_id": None, "score": None,
                      "rejected_candidate_ids": [candidate.candidate_id for candidate in candidates],
                      "reason": "no_safety_valid_candidate"}
    safe.sort(key=lambda candidate: (-candidate_score(candidate), len(candidate.steps), candidate.candidate_id))
    selected = safe[0]
    rejected_ids = [candidate.candidate_id for candidate in candidates if not candidate.safety_ok]
    summary = {
        "selected_candidate_id": selected.candidate_id,
        "score": round(candidate_score(selected), 6),
        "step_count": len(selected.steps),
        "rejected_candidate_ids": rejected_ids,
        "reason": "highest_score_safety_valid_plan",
        "candidate_summaries": [
            {"candidate_id": item.candidate_id, "score": round(candidate_score(item), 6),
             "step_count": len(item.steps), "safety_ok": item.safety_ok}
            for item in sorted(candidates, key=lambda item: item.candidate_id)
        ],
    }
    return selected, summary
