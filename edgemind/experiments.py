import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .models import RetrievalConfig


@dataclass(frozen=True)
class QualityGates:
    required_min: dict[str, float]
    max_regression: dict[str, float]


class ExperimentService:
    def __init__(self, store, evaluation_runner, gates: QualityGates):
        self.store = store
        self.evaluation_runner = evaluation_runner
        self.gates = gates

    def evaluate_candidate(self, config: RetrievalConfig, dataset_path) -> dict[str, Any]:
        baseline = self.store.get_active_version()
        if baseline is None:
            raise ValueError("an active baseline config version is required")
        evaluation = self.evaluation_runner.run(dataset_path, config=config, config_version="candidate")
        metrics = evaluation.get("metrics", {})
        baseline_metrics = baseline.get("metrics", {})
        metric_names = set(self.gates.required_min) | set(self.gates.max_regression)
        metric_diff = {
            name: (metrics[name] - baseline_metrics[name]
                   if isinstance(metrics.get(name), (int, float)) and isinstance(baseline_metrics.get(name), (int, float))
                   else None)
            for name in metric_names
        }
        reasons = []
        for name, minimum in self.gates.required_min.items():
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or value < minimum:
                reasons.append(f"{name} missing or below minimum {minimum}")
        for name, maximum_drop in self.gates.max_regression.items():
            value = metrics.get(name)
            base_value = baseline_metrics.get(name)
            if not isinstance(value, (int, float)) or not isinstance(base_value, (int, float)):
                reasons.append(f"{name} or baseline metric missing for regression gate")
            elif value - base_value < -maximum_drop:
                reasons.append(f"{name} regression exceeds {maximum_drop}")
        experiment_id = str(uuid.uuid4())
        result = {
            "experiment_id": experiment_id,
            "status": "passed" if not reasons else "blocked",
            "config": asdict(config),
            "baseline_version": baseline["version_id"],
            "metrics": metrics,
            "metric_diff": metric_diff,
            "gate_reasons": reasons,
            "evaluation": {key: value for key, value in evaluation.items() if key != "samples"},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.store.record_experiment(experiment_id, result)
        return result

    def promote(self, experiment_id: str) -> str:
        experiment = self.store.get_experiment(experiment_id)
        if experiment is None or experiment.get("status") != "passed":
            raise ValueError("experiment is not promotable")
        version_id = f"cfg-{uuid.uuid4().hex[:12]}"
        payload = {
            "config": experiment["config"],
            "metrics": experiment["metrics"],
            "parent_version": experiment["baseline_version"],
            "experiment_id": experiment_id,
        }
        self.store.record_version(version_id, payload)
        self.store.set_active_version(version_id)
        experiment.update({"status": "promoted", "version_id": version_id,
                           "promoted_at": datetime.now(timezone.utc).isoformat()})
        self.store.update_experiment(experiment_id, experiment)
        return version_id

    def rollback(self, version_id: str) -> str:
        current = self.store.get_active_version()
        self.store.set_active_version(version_id)
        if current and current.get("experiment_id"):
            experiment = self.store.get_experiment(current["experiment_id"])
            if experiment is not None:
                experiment.update({"status": "rolled_back", "rollback_to": version_id,
                                   "rolled_back_at": datetime.now(timezone.utc).isoformat()})
                self.store.update_experiment(current["experiment_id"], experiment)
        return version_id
