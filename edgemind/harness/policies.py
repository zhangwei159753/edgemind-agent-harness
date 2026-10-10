"""Versioned, declarative Agent policy snapshots with strict allowlists."""
from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Callable, Mapping

from .store import HarnessStore

BUILTIN_TOOL_IDS = frozenset({"edgemind__get_weather", "edgemind__web_search", "edgemind__fetch_search_result"})
MEMORY_LAYERS = ("working", "episodic", "semantic", "procedural")
CHECKER_IDS = frozenset({"expected_outcome", "allowlist", "tool_success", "memory_relevance", "citation_coverage"})
PROMPT_TEMPLATE_REFS = frozenset({"default-v1"})
_DEFAULT_PAYLOAD = {
    "prompt_suffix": "", "prompt_template_ref": None, "memory_top_k": 8,
    "layer_weights": {"working": 1.15, "episodic": 1.0, "semantic": 1.1, "procedural": 1.0},
    "time_decay_days": None, "allowed_memory_tags": [], "excluded_memory_tags": [],
    "tool_allowlist": sorted(BUILTIN_TOOL_IDS), "max_tool_rounds": 3,
    "output_constraints": {"require_citations": False, "max_output_chars": 6000}, "checker_ids": [],
}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bauthorization\s*[:=]\s*(?:basic|bearer)\s+[A-Za-z0-9+/=_-]{8,}"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\btoken\s*[:=]\s*\S{8,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
)
_FORBIDDEN_KEYS = {"python", "shell", "bash", "code", "exec", "executable", "command", "script", "module", "import"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _secret_or_code(value, path="payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden policy field at {path}")
            _secret_or_code(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _secret_or_code(child, path)
    elif isinstance(value, str) and any(pattern.search(value) for pattern in _SECRET_PATTERNS):
        raise ValueError(f"credential-like value at {path}")


def _secret_in_text(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


@dataclass(frozen=True)
class AgentPolicy:
    policy_id: str
    version: int
    prompt_suffix: str
    prompt_template_ref: str | None
    memory_top_k: int
    layer_weights: Mapping[str, float]
    time_decay_days: float | None
    allowed_memory_tags: tuple[str, ...]
    excluded_memory_tags: tuple[str, ...]
    tool_allowlist: tuple[str, ...]
    max_tool_rounds: int
    output_constraints: Mapping[str, object]
    checker_ids: tuple[str, ...]
    parent_policy_id: str | None


class AgentPolicyService:
    def __init__(self, store: HarnessStore, registered_tool_ids: Callable[[], set[str]] | None = None):
        self.store = store
        self.registered_tool_ids = registered_tool_ids or (lambda: set(BUILTIN_TOOL_IDS))

    @staticmethod
    def _from_record(record: dict) -> AgentPolicy:
        payload = record["payload"]
        return AgentPolicy(
            record["policy_id"], record["version"], payload["prompt_suffix"], payload["prompt_template_ref"],
            payload["memory_top_k"], _freeze(payload["layer_weights"]), payload["time_decay_days"],
            tuple(payload["allowed_memory_tags"]), tuple(payload["excluded_memory_tags"]),
            tuple(payload["tool_allowlist"]), payload["max_tool_rounds"], _freeze(payload["output_constraints"]),
            tuple(payload["checker_ids"]), record["parent_policy_id"],
        )

    def _baseline(self) -> AgentPolicy:
        tools = tuple(sorted(set(self.registered_tool_ids())))
        return AgentPolicy("baseline-v1", 1, "", None, 8, _freeze(_DEFAULT_PAYLOAD["layer_weights"]),
                           None, (), (), tools, 3, _freeze(_DEFAULT_PAYLOAD["output_constraints"]), (), None)

    def get_active(self) -> AgentPolicy:
        policy_id = self.store.active_agent_policy_id()
        if policy_id is None or policy_id == "baseline-v1":
            return self._baseline()
        return self.get(policy_id) or self._baseline()

    def get(self, policy_id: str) -> AgentPolicy | None:
        if policy_id == "baseline-v1":
            return self._baseline()
        record = self.store.get_agent_policy(policy_id)
        return None if record is None else self._from_record(record)

    def list_policies(self) -> list[dict]:
        active_id = self.get_active().policy_id
        result = [{"policy_id": "baseline-v1", "version": 1, "parent_policy_id": None,
                   "rationale": "Built-in immutable baseline", "active": active_id == "baseline-v1"}]
        for row in self.store.list_agent_policies():
            item = {key: row[key] for key in ("policy_id", "version", "parent_policy_id", "rationale", "source_run_id", "created_at")}
            item["active"] = active_id == row["policy_id"]
            item["payload"] = row["payload"]
            result.append(item)
        return result

    @staticmethod
    def _payload_from_policy(policy: AgentPolicy) -> dict:
        return {"prompt_suffix": policy.prompt_suffix, "prompt_template_ref": policy.prompt_template_ref,
                "memory_top_k": policy.memory_top_k, "layer_weights": dict(policy.layer_weights),
                "time_decay_days": policy.time_decay_days, "allowed_memory_tags": list(policy.allowed_memory_tags),
                "excluded_memory_tags": list(policy.excluded_memory_tags), "tool_allowlist": list(policy.tool_allowlist),
                "max_tool_rounds": policy.max_tool_rounds, "output_constraints": dict(policy.output_constraints),
                "checker_ids": list(policy.checker_ids)}

    def create_candidate(self, payload: dict, rationale: str, source_run_id: str | None = None) -> AgentPolicy:
        if not isinstance(payload, dict):
            raise ValueError("policy payload must be an object")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1000:
            raise ValueError("policy rationale must be non-empty and at most 1,000 characters")
        if _secret_in_text(rationale):
            raise ValueError("policy rationale appears to contain a credential")
        _secret_or_code(payload)
        unknown = set(payload) - set(_DEFAULT_PAYLOAD)
        if unknown:
            raise ValueError("unknown policy fields: " + ", ".join(sorted(unknown)))
        merged = self._payload_from_policy(self.get_active())
        merged.update(payload)
        self._validate(merged, set(self.registered_tool_ids()))
        if source_run_id is not None and self.store.find_agent_turn_event(source_run_id) is None:
            raise ValueError("source run does not exist")
        rows = self.store.list_agent_policies()
        version = max([1, *(row["version"] for row in rows)]) + 1
        parent = self.get_active()
        record = {"policy_id": str(uuid.uuid4()), "version": version, "parent_policy_id": parent.policy_id,
                  "payload": merged, "rationale": rationale.strip(), "source_run_id": source_run_id, "created_at": _now()}
        self.store.create_agent_policy(record)
        return self._from_record(record)

    @staticmethod
    def _validate(payload: dict, registered_tool_ids: set[str]) -> None:
        suffix = payload["prompt_suffix"]
        if not isinstance(suffix, str) or len(suffix) > 4000:
            raise ValueError("prompt_suffix must be text no longer than 4,000 characters")
        ref = payload["prompt_template_ref"]
        if ref is not None and (not isinstance(ref, str) or ref not in PROMPT_TEMPLATE_REFS):
            raise ValueError("prompt template reference is not registered")
        if ref is not None and suffix:
            raise ValueError("choose either prompt_suffix or prompt_template_ref")
        top_k = payload["memory_top_k"]
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 0 <= top_k <= 12:
            raise ValueError("memory_top_k must be between 0 and 12")
        weights = payload["layer_weights"]
        if not isinstance(weights, dict) or set(weights) != set(MEMORY_LAYERS):
            raise ValueError("layer_weights must specify exactly the four memory layers")
        for value in weights.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2:
                raise ValueError("layer weights must be finite numbers between 0 and 2")
        decay = payload["time_decay_days"]
        if decay is not None and (isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 1 <= decay <= 3650):
            raise ValueError("time_decay_days must be null or between 1 and 3,650")
        for name in ("allowed_memory_tags", "excluded_memory_tags"):
            values = payload[name]
            if not isinstance(values, list) or len(values) > 50 or any(
                not isinstance(item, str) or not item.strip() or len(item) > 64 for item in values
            ) or len(values) != len(set(values)):
                raise ValueError(f"{name} must be a list of at most 50 unique short tags")
        if set(payload["allowed_memory_tags"]) & set(payload["excluded_memory_tags"]):
            raise ValueError("memory tags cannot be both allowed and excluded")
        tools = payload["tool_allowlist"]
        if not isinstance(tools, list) or any(not isinstance(item, str) for item in tools) or len(tools) != len(set(tools)):
            raise ValueError("tool_allowlist must be a list of unique tool IDs")
        if any(item not in registered_tool_ids for item in tools):
            raise ValueError("tool allowlist contains a tool that is not registered")
        rounds = payload["max_tool_rounds"]
        if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 3:
            raise ValueError("max_tool_rounds must be between 1 and 3")
        constraints = payload["output_constraints"]
        if not isinstance(constraints, dict) or set(constraints) != {"require_citations", "max_output_chars"}:
            raise ValueError("output_constraints has an unsupported schema")
        if not isinstance(constraints["require_citations"], bool):
            raise ValueError("require_citations must be boolean")
        chars = constraints["max_output_chars"]
        if not isinstance(chars, int) or isinstance(chars, bool) or not 128 <= chars <= 10000:
            raise ValueError("max_output_chars must be between 128 and 10,000")
        checkers = payload["checker_ids"]
        if not isinstance(checkers, list) or any(not isinstance(item, str) for item in checkers) or len(checkers) != len(set(checkers)):
            raise ValueError("checker_ids must be a list of unique checker IDs")
        if any(item not in CHECKER_IDS for item in checkers):
            raise ValueError("checker_ids contains an unsupported checker")

    def promote(self, policy_id: str, experiment_id: str, confirmed: bool) -> AgentPolicy:
        if not confirmed:
            raise ValueError("policy promotion requires explicit confirmation")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("promotion requires an experiment ID")
        policy = self.get(policy_id)
        if policy is None or policy_id == "baseline-v1":
            raise ValueError("candidate policy does not exist")
        self.store.activate_agent_policy(policy_id, "promote", experiment_id.strip())
        return policy

    def rollback(self, policy_id: str, confirmed: bool) -> AgentPolicy:
        if not confirmed:
            raise ValueError("policy rollback requires explicit confirmation")
        policy = self.get(policy_id)
        if policy is None:
            raise ValueError("rollback target policy does not exist")
        self.store.activate_agent_policy(policy_id, "rollback")
        return policy
