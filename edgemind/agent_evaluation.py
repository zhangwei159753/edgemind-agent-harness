"""Fixture-first, isolated evaluation of long-running Agent behavior."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .harness.agent import AgentService
from .harness.models import MemoryItem
from .harness.policies import AgentPolicy, CHECKER_IDS
from .harness.store import HarnessStore
from .harness.tasks import TaskService

_MAX_DATASET_BYTES = 2_000_000
_MAX_CASE_BYTES = 64_000
_MAX_CASES = 100
_ALLOWED_CASE_KEYS = {"record_type", "example_id", "task", "input", "available_tools", "memory_fixtures",
                      "fixture", "expected", "checker_ids"}
_ALLOWED_TASK_KEYS = {"title", "objective", "state"}
_ALLOWED_EXPECTED_KEYS = {"answer_contains", "tool_names", "forbidden_tool_names", "task_outcome", "memory_ids",
                          "citation_ids", "tool_success"}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bauthorization\s*[:=]\s*(?:basic|bearer)\s+[A-Za-z0-9+/=_-]{8,}"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|token)\s*[:=]\s*\S{8,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
)
_FORBIDDEN_KEYS = {"python", "shell", "bash", "code", "exec", "executable", "command", "script", "module", "import"}
_SENSITIVE_KEYS = {"authorization", "proxyauthorization", "apikey", "accesstoken", "clientsecret",
                   "password", "passwd", "secret"}


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _check_safe(value: Any, path: str = "dataset") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower()) if isinstance(key, str) else ""
            if normalized_key in _SENSITIVE_KEYS:
                raise ValueError(f"credential field is forbidden at {path}")
            if not isinstance(key, str) or key.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"dynamic-code field is forbidden at {path}")
            _check_safe(item, f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            _check_safe(item, path)
    elif isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
            raise ValueError(f"credential-like value is forbidden at {path}")


@dataclass(frozen=True)
class AgentDatasetCase:
    example_id: str
    task: Mapping[str, Any]
    input: str
    available_tools: tuple[Any, ...]
    memory_fixtures: tuple[Mapping[str, Any], ...]
    fixture: Mapping[str, Any]
    expected: Mapping[str, Any]
    checker_ids: tuple[str, ...]


@dataclass(frozen=True)
class AgentDataset:
    dataset_id: str
    version: int
    description: str
    cases: tuple[AgentDatasetCase, ...]
    dataset_hash: str
    path: str


def load_agent_dataset(path: str | Path) -> AgentDataset:
    dataset_path = Path(path)
    try:
        raw_bytes = dataset_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read Agent dataset: {exc}") from exc
    if not raw_bytes or len(raw_bytes) > _MAX_DATASET_BYTES:
        raise ValueError("Agent dataset must be non-empty and no larger than 2 MB")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Agent dataset must be UTF-8 JSONL") from exc
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > _MAX_CASE_BYTES:
            raise ValueError(f"dataset record on line {line_number} exceeds 64 KB")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on dataset line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"dataset record on line {line_number} must be an object")
        _check_safe(record, f"line {line_number}")
        records.append(record)
    if len(records) < 2 or records[0].get("record_type") != "manifest":
        raise ValueError("the first JSONL record must be a dataset manifest")
    manifest = records[0]
    if set(manifest) != {"record_type", "dataset_id", "version", "description"}:
        raise ValueError("invalid dataset manifest fields")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"].strip():
        raise ValueError("dataset_id is required")
    if manifest["version"] != 1 or isinstance(manifest["version"], bool):
        raise ValueError("unsupported Agent dataset schema version")
    if not isinstance(manifest["description"], str) or len(manifest["description"]) > 1000:
        raise ValueError("dataset description is invalid")
    if len(records) - 1 > _MAX_CASES:
        raise ValueError("Agent dataset may contain at most 100 cases")
    cases, seen = [], set()
    for index, raw in enumerate(records[1:], 2):
        required = _ALLOWED_CASE_KEYS
        if set(raw) != required or raw.get("record_type") != "case":
            raise ValueError(f"invalid case fields on line {index}")
        example_id = raw.get("example_id")
        if not isinstance(example_id, str) or not example_id.strip() or len(example_id) > 120:
            raise ValueError(f"invalid example_id on line {index}")
        if example_id in seen:
            raise ValueError(f"duplicate example_id: {example_id}")
        seen.add(example_id)
        task = raw.get("task")
        if not isinstance(task, dict) or set(task) - _ALLOWED_TASK_KEYS:
            raise ValueError(f"invalid task fixture for {example_id}")
        if not all(isinstance(task.get(key), str) and task[key].strip() for key in ("title", "objective")):
            raise ValueError(f"task title and objective are required for {example_id}")
        if len(task["title"]) > 200 or len(task["objective"]) > 2000:
            raise ValueError(f"task fixture is too long for {example_id}")
        if not isinstance(task.get("state", {}), dict):
            raise ValueError(f"task state must be an object for {example_id}")
        if not isinstance(raw.get("input"), str) or not raw["input"].strip() or len(raw["input"]) > 20000:
            raise ValueError(f"input must be 1–20,000 characters for {example_id}")
        tools = raw.get("available_tools")
        if not isinstance(tools, list) or len(tools) > 50:
            raise ValueError(f"available_tools must be a list for {example_id}")
        for tool in tools:
            name = tool if isinstance(tool, str) else tool.get("name") if isinstance(tool, dict) else None
            if not isinstance(name, str) or "__" not in name or len(name) > 120:
                raise ValueError(f"invalid available tool ID for {example_id}")
            if isinstance(tool, dict) and set(tool) != {"name", "side_effecting"}:
                raise ValueError(f"invalid fixture tool descriptor for {example_id}")
        memories = raw.get("memory_fixtures")
        if not isinstance(memories, list) or len(memories) > 50:
            raise ValueError(f"memory_fixtures must be a list for {example_id}")
        memory_ids = set()
        for memory in memories:
            if not isinstance(memory, dict) or set(memory) - {"memory_id", "layer", "content", "status", "metadata"}:
                raise ValueError(f"invalid memory fixture for {example_id}")
            if not all(isinstance(memory.get(key), str) and memory[key] for key in ("memory_id", "layer", "content")):
                raise ValueError(f"memory fixture is missing required fields for {example_id}")
            if memory["layer"] not in {"working", "episodic", "semantic", "procedural"}:
                raise ValueError(f"invalid memory layer for {example_id}")
            if memory.get("status", "approved") != "approved":
                raise ValueError("Agent evaluation memory fixtures must be approved")
            if memory["memory_id"] in memory_ids:
                raise ValueError(f"duplicate memory fixture ID in {example_id}")
            memory_ids.add(memory["memory_id"])
        fixture = raw.get("fixture")
        if not isinstance(fixture, dict) or set(fixture) - {"responses", "tool_outputs", "citations"}:
            raise ValueError(f"invalid fixture object for {example_id}")
        responses = fixture.get("responses")
        if not isinstance(responses, list) or not responses or len(responses) > 12:
            raise ValueError(f"fixture responses are required for {example_id}")
        for response in responses:
            if not isinstance(response, dict) or set(response) - {"content", "tool_calls"}:
                raise ValueError(f"invalid provider response fixture for {example_id}")
            calls = response.get("tool_calls", [])
            if not isinstance(calls, list) or len(calls) > 12:
                raise ValueError(f"invalid tool calls fixture for {example_id}")
            for call in calls:
                if not isinstance(call, dict) or set(call) - {"name", "arguments", "id"} or not isinstance(call.get("name"), str):
                    raise ValueError(f"invalid tool call fixture for {example_id}")
                if not isinstance(call.get("arguments", {}), dict):
                    raise ValueError(f"tool arguments fixture must be an object for {example_id}")
        expected = raw.get("expected")
        if not isinstance(expected, dict) or set(expected) - _ALLOWED_EXPECTED_KEYS:
            raise ValueError(f"invalid expected checks for {example_id}")
        checker_ids = raw.get("checker_ids")
        if not isinstance(checker_ids, list) or any(item not in CHECKER_IDS for item in checker_ids):
            raise ValueError(f"unsupported checker for {example_id}")
        cases.append(AgentDatasetCase(example_id, _freeze(task), raw["input"], tuple(_freeze(tools)),
                                      tuple(_freeze(memories)), _freeze(fixture), _freeze(expected), tuple(checker_ids)))
    return AgentDataset(manifest["dataset_id"], manifest["version"], manifest["description"], tuple(cases),
                        hashlib.sha256(raw_bytes).hexdigest(), str(dataset_path.resolve()))


class AgentEvaluationRunner:
    """Runs fixture or explicitly confirmed live cases using isolated Harness stores."""

    def __init__(self, agent_factory: Callable, fixture_tool_factory: Callable,
                 temp_store_factory: Callable[[], HarnessStore]):
        self.agent_factory = agent_factory
        self.fixture_tool_factory = fixture_tool_factory
        self.temp_store_factory = temp_store_factory

    def run(self, dataset_path: str | Path, policy: AgentPolicy, mode: str = "fixture",
            repetitions: int = 1, confirmed_live: bool = False) -> dict[str, Any]:
        if mode not in {"fixture", "live"}:
            raise ValueError("mode must be fixture or live")
        if mode == "live" and not confirmed_live:
            raise ValueError("live evaluation requires explicit confirmation")
        if not isinstance(repetitions, int) or isinstance(repetitions, bool) or not 1 <= repetitions <= 5:
            raise ValueError("repetitions must be between 1 and 5")
        dataset = load_agent_dataset(dataset_path)
        all_case_results, repetition_metrics, stores = [], [], []
        for repetition in range(1, repetitions + 1):
            cases = []
            for case in dataset.cases:
                try:
                    required_tools = set(case.expected.get("tool_names", ()))
                    unavailable = sorted(required_tools - set(policy.tool_allowlist))
                    if unavailable:
                        cases.append({"example_id": case.example_id, "repetition": repetition,
                                      "mode": mode, "passed": None, "actual_status": "not_applicable",
                                      "na_reason": "policy does not allow required tool(s): " + ", ".join(unavailable),
                                      "checks": {}, "tool_calls": [], "unsafe_tool_attempts": 0})
                        continue
                    store = self.temp_store_factory()
                    self._assert_isolated_store(store, stores)
                    task = TaskService(store).create(case.task["title"], case.task["objective"])
                    if case.task.get("state"):
                        store.update_task_state(task.task_id, state=dict(case.task["state"]))
                    for memory in case.memory_fixtures:
                        store.put_memory(MemoryItem(memory["memory_id"], memory["layer"], memory["content"], "approved",
                                                    task.task_id, None, 1.0, dict(memory.get("metadata", {}))))
                    fixtures = self.fixture_tool_factory(case) if mode == "fixture" else None
                    if mode == "fixture" and not isinstance(fixtures, dict):
                        raise ValueError("fixture_tool_factory must return a fixture bundle")
                    agent = self.agent_factory(store, policy, case, fixtures)
                    if not isinstance(agent, AgentService) or agent.store is not store:
                        raise ValueError("agent_factory must return AgentService bound to the isolated HarnessStore")
                    if mode == "fixture":
                        self._assert_fixture_wiring(agent, fixtures)
                    turn = agent.run_turn(task.task_id, case.input)
                    result = self._evaluate_case(case, turn, policy, mode, repetition)
                    result["task_id"] = task.task_id
                    cases.append(result)
                except Exception as exc:
                    cases.append({"example_id": case.example_id, "repetition": repetition, "mode": mode,
                                  "passed": False, "actual_status": "error", "error_category": type(exc).__name__,
                                  "checks": {"runner_error": False}})
            all_case_results.extend(cases)
            repetition_metrics.append(self._metrics(cases))
        metrics, na_reasons = self._aggregate_metrics(repetition_metrics, all_case_results)
        case_results = all_case_results
        applicable_case_count = sum(item.get("passed") is not None for item in case_results)
        policy_hash = self.policy_hash(policy)
        return {
            "report_id": str(uuid.uuid4()), "dataset_id": dataset.dataset_id, "dataset_version": dataset.version,
            "dataset_path": dataset.path, "dataset_hash": dataset.dataset_hash,
            "policy_id": policy.policy_id, "policy_version": policy.version, "policy_hash": policy_hash,
            "mode": mode, "provider": "fixture" if mode == "fixture" else "configured live provider",
            "model": "fixture-stub" if mode == "fixture" else self._first_model(case_results),
            "provider_metadata": {"provider": "fixture" if mode == "fixture" else "configured live provider",
                                  "usage_available": False}, "repetitions": repetitions,
            "case_count": len(dataset.cases), "applicable_case_count": applicable_case_count,
            "not_applicable_case_count": len(case_results) - applicable_case_count,
            "metrics": metrics, "na_reasons": na_reasons,
            "repetition_metrics": repetition_metrics, "repeated_metric_summary": self._repetition_summary(repetition_metrics),
            "case_results": case_results,
        }

    @staticmethod
    def _assert_isolated_store(store: HarnessStore, used_stores: list[HarnessStore]) -> None:
        if not isinstance(store, HarnessStore):
            raise ValueError("temp_store_factory must return HarnessStore")
        if any(store is previous for previous in used_stores):
            raise ValueError("evaluation store instance was reused")
        used_stores.append(store)
        path = Path(tempfile.gettempdir()).resolve()
        if store.db_path != ":memory:":
            try:
                Path(store.db_path).resolve().relative_to(path)
            except ValueError as exc:
                raise ValueError("evaluation store must live under the system temporary directory") from exc


    @staticmethod
    def _assert_fixture_wiring(agent: AgentService, fixtures: dict) -> None:
        required = {"client", "builtin_tools", "mcp_manager"}
        if not required.issubset(fixtures):
            raise ValueError("fixture bundle must provide client, builtin_tools and mcp_manager")
        if agent.client is not fixtures["client"] or agent.builtin_tools is not fixtures["builtin_tools"]:
            raise ValueError("fixture Agent must use the injected client and built-in tool fixtures")
        if agent.tools.manager is not fixtures["mcp_manager"]:
            raise ValueError("fixture Agent must use the injected MCP fixture; live MCP is forbidden")

    @staticmethod
    def policy_hash(policy: AgentPolicy) -> str:
        payload = {"policy_id": policy.policy_id, "version": policy.version, "prompt_suffix": policy.prompt_suffix,
                   "prompt_template_ref": policy.prompt_template_ref, "memory_top_k": policy.memory_top_k,
                   "layer_weights": dict(policy.layer_weights), "time_decay_days": policy.time_decay_days,
                   "allowed_memory_tags": list(policy.allowed_memory_tags), "excluded_memory_tags": list(policy.excluded_memory_tags),
                   "tool_allowlist": list(policy.tool_allowlist), "max_tool_rounds": policy.max_tool_rounds,
                   "output_constraints": dict(policy.output_constraints), "checker_ids": list(policy.checker_ids)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    @staticmethod
    def _evaluate_case(case: AgentDatasetCase, turn, policy: AgentPolicy, mode: str, repetition: int) -> dict:
        expected = dict(case.expected)
        selected_checkers = set(case.checker_ids) | set(policy.checker_ids)
        actual_tools = [entry.get("name") for entry in turn.trace.get("tool_calls", []) if isinstance(entry, dict)]
        tool_statuses = [entry.get("status") for entry in turn.trace.get("tool_calls", []) if isinstance(entry, dict)]
        forbidden = set(expected.get("forbidden_tool_names", []))
        checks = {}
        if "expected_outcome" in selected_checkers and "answer_contains" in expected:
            answer_lower = (turn.answer or "").lower()
            checks["answer_contains"] = all(str(item).lower() in answer_lower for item in expected["answer_contains"])
        if "allowlist" in selected_checkers and "tool_names" in expected:
            checks["tool_names"] = actual_tools == list(expected["tool_names"])
        if "allowlist" in selected_checkers and forbidden:
            checks["forbidden_tools_not_selected"] = not any(item in forbidden for item in actual_tools)
        if "expected_outcome" in selected_checkers and "task_outcome" in expected:
            actual_outcome = {"succeeded": "success", "failed": "failure",
                              "awaiting_tool_approval": "awaiting_tool_approval"}.get(turn.status, turn.status)
            checks["task_outcome"] = actual_outcome == expected["task_outcome"]
        actual_memories = {item.get("memory_id") for item in turn.trace.get("memory_hits", []) if isinstance(item, dict)}
        if "memory_relevance" in selected_checkers and "memory_ids" in expected:
            checks["memory_ids"] = set(expected["memory_ids"]).issubset(actual_memories)
        citations = AgentEvaluationRunner._citation_ids(turn.trace.get("rag_trace"))
        if "citation_coverage" in selected_checkers and "citation_ids" in expected:
            checks["citation_ids"] = set(expected["citation_ids"]).issubset(citations)
        if "tool_success" in selected_checkers and "tool_success" in expected:
            resolved = [status for status in tool_statuses if status != "awaiting_confirmation"]
            actual_success = bool(resolved) and all(status == "succeeded" for status in resolved)
            checks["tool_success"] = actual_success is expected["tool_success"]
        unauthorized = [call for call in turn.trace.get("tool_calls", [])
                        if isinstance(call, dict) and (
                            (call.get("status") == "failed" and call.get("error_type") == "tool_not_allowed")
                            or call.get("name") in forbidden
                        )]
        checks["runtime_allowlist"] = not unauthorized
        passed = all(checks.values()) if checks else True
        checker_expectations = {
            "expected_outcome": ("task_outcome", "answer_contains"), "allowlist": ("tool_names", "forbidden_tool_names"),
            "tool_success": ("tool_success",), "memory_relevance": ("memory_ids",),
            "citation_coverage": ("citation_ids",),
        }
        checker_na = [checker for checker in sorted(selected_checkers)
                      if not any(key in expected for key in checker_expectations[checker])]
        return {"example_id": case.example_id, "repetition": repetition, "mode": mode,
                "passed": passed, "actual_status": turn.status, "answer_preview": _redact(turn.answer or "")[:500],
                "tool_calls": [{"name": item.get("name"), "status": item.get("status"),
                                "error_type": item.get("error_type")}
                               for item in turn.trace.get("tool_calls", []) if isinstance(item, dict)],
                "memory_ids": sorted(actual_memories), "expected_memory_ids": list(expected.get("memory_ids", [])),
                "citation_ids": sorted(citations), "expected_citation_ids": list(expected.get("citation_ids", [])),
                "expected_outcome": expected.get("task_outcome"),
                "actual_outcome": {"succeeded": "success", "failed": "failure",
                                   "awaiting_tool_approval": "awaiting_tool_approval"}.get(turn.status, turn.status),
                "latency_ms": turn.trace.get("elapsed_ms"), "model": turn.trace.get("model"),
                "checks": checks, "checker_na": checker_na,
                "error_category": turn.trace.get("error_type") if turn.status == "failed" else None,
                "unsafe_tool_attempts": len(unauthorized)}

    @staticmethod
    def _citation_ids(rag_trace) -> set[str]:
        if not isinstance(rag_trace, dict):
            return set()
        citations = rag_trace.get("citations", [])
        result = set()
        for item in citations if isinstance(citations, (list, tuple)) else []:
            if isinstance(item, str):
                result.add(item)
            elif isinstance(item, dict):
                for key in ("citation_id", "chunk_id", "id"):
                    if isinstance(item.get(key), str):
                        result.add(item[key])
                        break
        return result

    @staticmethod
    def _metrics(cases: list[dict]) -> dict[str, float | int | None]:
        applicable = [case for case in cases if case.get("passed") is not None]
        tool_calls = [call for case in applicable for call in case.get("tool_calls", [])]
        completed_calls = [call for call in tool_calls if call["status"] != "awaiting_confirmation"]
        labeled_outcomes = [case for case in applicable if case.get("expected_outcome") is not None]
        memory_cases = [case for case in applicable if case.get("expected_memory_ids")]
        memory_tp = sum(len(set(case["memory_ids"]) & set(case["expected_memory_ids"])) for case in memory_cases)
        memory_retrieved = sum(len(case["memory_ids"]) for case in memory_cases)
        memory_gold = sum(len(case["expected_memory_ids"]) for case in memory_cases)
        citation_cases = [case for case in applicable if case.get("expected_citation_ids")]
        citation_tp = sum(len(set(case["citation_ids"]) & set(case["expected_citation_ids"])) for case in citation_cases)
        citation_gold = sum(len(case["expected_citation_ids"]) for case in citation_cases)
        latencies = [case["latency_ms"] for case in applicable if isinstance(case.get("latency_ms"), (int, float))]
        attempts = len(tool_calls)
        safe_calls = attempts - sum(case.get("unsafe_tool_attempts", 0) for case in cases)
        return {
            "case_pass_rate": sum(bool(case.get("passed")) for case in applicable) / len(applicable) if applicable else None,
            "allowlist_compliance": safe_calls / attempts if attempts else None,
            "forbidden_tool_attempts": sum(case.get("unsafe_tool_attempts", 0) for case in cases),
            "tool_call_success_rate": sum(call["status"] == "succeeded" for call in completed_calls) / len(completed_calls) if completed_calls else None,
            "expected_outcome_accuracy": (sum(case["expected_outcome"] == case["actual_outcome"] for case in labeled_outcomes) / len(labeled_outcomes)) if labeled_outcomes else None,
            "memory_precision_at_k": memory_tp / memory_retrieved if memory_cases and memory_retrieved else (0.0 if memory_cases else None),
            "memory_recall_at_k": memory_tp / memory_gold if memory_gold else None,
            "citation_coverage": citation_tp / citation_gold if citation_gold else None,
            "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
            "user_feedback_outcome_rate": None,
            "token_usage": None,
        }

    @staticmethod
    def _aggregate_metrics(repetitions: list[dict], cases: list[dict]) -> tuple[dict, dict]:
        names = set().union(*(metrics.keys() for metrics in repetitions)) if repetitions else set()
        metrics, reasons = {}, {}
        for name in names:
            values = [item[name] for item in repetitions if isinstance(item.get(name), (int, float))]
            if values:
                metrics[name] = statistics.fmean(values)
            else:
                metrics[name] = None
                reasons[name] = {
                    "user_feedback_outcome_rate": "evaluation fixtures do not fabricate user feedback",
                    "token_usage": "provider usage is unavailable in the injected fixture responses",
                    "tool_call_success_rate": "no tool execution occurred",
                    "allowlist_compliance": "no tool calls occurred",
                    "expected_outcome_accuracy": "no dataset outcome labels are available",
                    "memory_precision_at_k": "no cases provide gold memory IDs",
                    "memory_recall_at_k": "no cases provide gold memory IDs",
                    "citation_coverage": "no cases provide required citation IDs",
                }.get(name, "no labeled data is available")
        return metrics, reasons

    @staticmethod
    def _repetition_summary(repetitions: list[dict]) -> dict:
        summary = {}
        for name in set().union(*(item.keys() for item in repetitions)) if repetitions else set():
            values = [item[name] for item in repetitions if isinstance(item.get(name), (int, float))]
            if values:
                summary[name] = {"mean": statistics.fmean(values), "dispersion": statistics.pstdev(values) if len(values) > 1 else 0.0}
            else:
                summary[name] = {"mean": None, "dispersion": None}
        return summary

    @staticmethod
    def _first_model(cases: list[dict]) -> str | None:
        return next((case["model"] for case in cases if case.get("model")), None)


def _redact(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value
