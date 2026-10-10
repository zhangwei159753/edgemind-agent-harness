"""Quality gates and human-confirmed Agent policy experiments."""
from __future__ import annotations

import re
import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .agent_evaluation import AgentEvaluationRunner
from .harness.policies import AgentPolicyService
from .harness.store import HarnessStore

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bauthorization\s*[:=]\s*(?:basic|bearer)\s+[A-Za-z0-9+/=_-]{8,}"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|token)\s*[:=]\s*\S{8,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value[:500]


@dataclass(frozen=True)
class AgentQualityGates:
    required_min: Mapping[str, float] = field(default_factory=dict)
    max_regression: Mapping[str, float] = field(default_factory=dict)
    max_latency_ms: float | None = None

    def __post_init__(self):
        for name, minimum in self.required_min.items():
            if not isinstance(name, str) or isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(minimum):
                raise ValueError("required_min must map metric names to numbers")
        for name, tolerance in self.max_regression.items():
            if not isinstance(name, str) or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("max_regression must map metric names to non-negative numbers")
        if self.max_latency_ms is not None and (
            isinstance(self.max_latency_ms, bool) or not isinstance(self.max_latency_ms, (int, float)) or not math.isfinite(self.max_latency_ms) or self.max_latency_ms <= 0
        ):
            raise ValueError("max_latency_ms must be a positive number")


class AgentExperimentService:
    def __init__(self, policy_service: AgentPolicyService, evaluation_runner: AgentEvaluationRunner,
                 store: HarnessStore, gates: AgentQualityGates):
        self.policy_service = policy_service
        self.evaluation_runner = evaluation_runner
        self.store = store
        self.gates = gates

    def evaluate_candidate(self, policy_id: str, dataset_path: str | Path,
                           mode: str = "fixture", repetitions: int = 1,
                           confirmed_live: bool = False) -> dict:
        if mode not in {"fixture", "live"}:
            raise ValueError("mode must be fixture or live")
        if mode == "live" and not confirmed_live:
            raise ValueError("live candidate evaluation requires explicit confirmation")
        candidate = self.policy_service.get(policy_id)
        if candidate is None or policy_id == "baseline-v1":
            raise ValueError("candidate policy does not exist")
        baseline = self.policy_service.get(candidate.parent_policy_id or "baseline-v1")
        if baseline is None:
            raise ValueError("candidate parent policy does not exist")
        experiment_id = str(uuid.uuid4())
        baseline_report = candidate_report = None
        error_category = None
        error_message = None
        try:
            baseline_report = self.evaluation_runner.run(dataset_path, baseline, mode=mode,
                                                        repetitions=repetitions, confirmed_live=confirmed_live)
            candidate_report = self.evaluation_runner.run(dataset_path, candidate, mode=mode,
                                                          repetitions=repetitions, confirmed_live=confirmed_live)
        except Exception as exc:
            error_category = type(exc).__name__
            error_message = _redact(str(exc))
        reasons = self._gate_reasons(baseline_report, candidate_report, error_category)
        dataset_hash = (candidate_report or {}).get("dataset_hash") or (baseline_report or {}).get("dataset_hash") or ""
        report = {
            "experiment_id": experiment_id, "candidate_policy_id": candidate.policy_id,
            "baseline_policy_id": baseline.policy_id, "dataset_path": str(dataset_path),
            "dataset_hash": dataset_hash, "mode": mode, "repetitions": repetitions,
            "baseline": baseline_report, "candidate": candidate_report,
            "metrics_delta": self._metrics_delta(baseline_report, candidate_report),
            "case_differences": self._case_differences(baseline_report, candidate_report),
            "passed": not reasons, "gate_reasons": reasons,
            "error_category": error_category, "error_message": error_message, "created_at": _now(),
        }
        self.store.save_agent_experiment(report)
        return report

    def _gate_reasons(self, baseline: dict | None, candidate: dict | None,
                      error_category: str | None) -> list[str]:
        reasons = []
        if error_category:
            reasons.append(f"Evaluator failed: {error_category}")
        if baseline is None or candidate is None:
            reasons.append("Baseline and candidate reports are both required")
            return reasons
        if not baseline.get("dataset_hash") or baseline.get("dataset_hash") != candidate.get("dataset_hash"):
            reasons.append("Baseline and candidate must use the same versioned dataset hash")
        if baseline.get("mode") != candidate.get("mode"):
            reasons.append("Baseline and candidate evaluation modes differ")
        for label, report in (("baseline", baseline), ("candidate", candidate)):
            for case in report.get("case_results", []):
                if not isinstance(case, dict) or case.get("passed") is None:
                    continue
                if case.get("actual_status") == "error" or case.get("error_category"):
                    reasons.append(f"{label} evaluator error in case {case.get('example_id', 'unknown')}")
                elif case.get("actual_status") == "failed" and case.get("expected_outcome") != "failure":
                    reasons.append(f"{label} failed case {case.get('example_id', 'unknown')} unexpectedly")
                if case.get("unsafe_tool_attempts", 0):
                    reasons.append(f"{label} unsafe tool attempt in case {case.get('example_id', 'unknown')}")
        base_metrics = baseline.get("metrics") or {}
        candidate_metrics = candidate.get("metrics") or {}
        unsafe = candidate_metrics.get("forbidden_tool_attempts")
        if not isinstance(unsafe, (int, float)) or unsafe != 0:
            reasons.append("Forbidden or unauthorized tool attempts must be zero")
        for name, minimum in self.gates.required_min.items():
            value = candidate_metrics.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                reasons.append(f"Required metric {name} is N/A")
            elif value < minimum:
                reasons.append(f"Required metric {name}={value:g} is below {minimum:g}")
        for name, tolerance in self.gates.max_regression.items():
            base, current = base_metrics.get(name), candidate_metrics.get(name)
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in (base, current)):
                reasons.append(f"Regression metric {name} is N/A for baseline or candidate")
            elif current < base - tolerance:
                reasons.append(f"Metric {name} regressed from {base:g} to {current:g}, tolerance {tolerance:g}")
        if self.gates.max_latency_ms is not None:
            latency = candidate_metrics.get("latency_ms_mean")
            if not isinstance(latency, (int, float)) or isinstance(latency, bool):
                reasons.append("Latency metric is N/A")
            elif latency > self.gates.max_latency_ms:
                reasons.append(f"Latency {latency:g} ms exceeds {self.gates.max_latency_ms:g} ms budget")
        return reasons

    @staticmethod
    def _metrics_delta(baseline: dict | None, candidate: dict | None) -> dict:
        if not baseline or not candidate:
            return {}
        base, current = baseline.get("metrics", {}), candidate.get("metrics", {})
        return {name: current[name] - base[name] for name in set(base) & set(current)
                if isinstance(base[name], (int, float)) and not isinstance(base[name], bool)
                and isinstance(current[name], (int, float)) and not isinstance(current[name], bool)}

    @staticmethod
    def _case_differences(baseline: dict | None, candidate: dict | None) -> list[dict]:
        if not baseline or not candidate:
            return []
        def index(report):
            return {(item.get("example_id"), item.get("repetition")): item
                    for item in report.get("case_results", []) if isinstance(item, dict)}
        base, current = index(baseline), index(candidate)
        differences = []
        for key in sorted(set(base) | set(current), key=str):
            left, right = base.get(key), current.get(key)
            if left is None or right is None or left.get("passed") != right.get("passed") or left.get("checks") != right.get("checks"):
                differences.append({"example_id": key[0], "repetition": key[1],
                                    "baseline_passed": None if left is None else left.get("passed"),
                                    "candidate_passed": None if right is None else right.get("passed"),
                                    "baseline_checks": None if left is None else left.get("checks"),
                                    "candidate_checks": None if right is None else right.get("checks")})
        return differences

    def promote(self, experiment_id: str, confirmed: bool):
        if not confirmed:
            raise ValueError("policy promotion requires explicit confirmation")
        experiment = self.store.get_agent_experiment(experiment_id)
        if experiment is None or not experiment["passed"]:
            raise ValueError("promotion is blocked because the experiment gate did not pass")
        return self.policy_service.promote(experiment["candidate_policy_id"], experiment_id, confirmed=True)

    def rollback(self, policy_id: str, confirmed: bool):
        return self.policy_service.rollback(policy_id, confirmed=confirmed)
