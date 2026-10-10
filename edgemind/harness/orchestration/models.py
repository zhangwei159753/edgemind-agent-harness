from dataclasses import dataclass, field
from typing import Any, Mapping


ROLE_SCHEMAS = {
    "planner": "planner-v1",
    "researcher": "researcher-v1",
    "executor": "executor-v1",
    "reviewer": "reviewer-v1",
}
ROLE_STATUSES = {"succeeded", "failed", "rejected", "awaiting_tool_approval", "skipped"}
_MAX_COLLECTION = 50
_MAX_CALLS = 32


def _bounded_text(value: Any, field: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be non-empty text no longer than {maximum} characters")
    return value


def _string_tuple(value: Any, field: str, *, maximum: int = _MAX_COLLECTION, item_length: int = 200) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) > maximum:
        raise ValueError(f"{field} must be a list or tuple of at most {maximum} items")
    result = tuple(_bounded_text(item, field, item_length) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True)
class Handoff:
    run_id: str
    task_id: str
    role: str
    goal: str
    evidence_ids: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    remaining_calls: int
    output_schema: str

    def __post_init__(self) -> None:
        _bounded_text(self.run_id, "run_id", 120)
        _bounded_text(self.task_id, "task_id", 120)
        if self.role not in ROLE_SCHEMAS:
            raise ValueError("unsupported role")
        _bounded_text(self.goal, "goal", 4000)
        evidence_ids = _string_tuple(self.evidence_ids, "evidence_ids")
        allowed_tools = _string_tuple(self.allowed_tools, "allowed_tools", item_length=120)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "allowed_tools", allowed_tools)
        if (not isinstance(self.remaining_calls, int) or isinstance(self.remaining_calls, bool)
                or not 1 <= self.remaining_calls <= _MAX_CALLS):
            raise ValueError(f"remaining_calls must be an integer from 1 to {_MAX_CALLS}")
        if self.output_schema != ROLE_SCHEMAS[self.role]:
            raise ValueError("unsupported output schema for role")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Handoff":
        if not isinstance(payload, Mapping):
            raise ValueError("handoff must be an object")
        required = {"run_id", "task_id", "role", "goal", "evidence_ids", "allowed_tools", "remaining_calls", "output_schema"}
        unknown = set(payload) - required
        missing = required - set(payload)
        if unknown:
            raise ValueError("handoff contains unknown fields: " + ", ".join(sorted(str(key) for key in unknown)))
        if missing:
            raise ValueError("handoff is missing fields: " + ", ".join(sorted(missing)))
        return cls(
            run_id=payload["run_id"], task_id=payload["task_id"], role=payload["role"],
            goal=payload["goal"], evidence_ids=payload["evidence_ids"],
            allowed_tools=payload["allowed_tools"], remaining_calls=payload["remaining_calls"],
            output_schema=payload["output_schema"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "task_id": self.task_id, "role": self.role,
            "goal": self.goal, "evidence_ids": list(self.evidence_ids),
            "allowed_tools": list(self.allowed_tools), "remaining_calls": self.remaining_calls,
            "output_schema": self.output_schema,
        }


@dataclass(frozen=True)
class RoleResult:
    role: str
    status: str
    payload: dict[str, Any]
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLE_SCHEMAS:
            raise ValueError("unsupported role")
        if self.status not in ROLE_STATUSES:
            raise ValueError("unsupported status")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be an object")
        if self.error_type is not None:
            _bounded_text(self.error_type, "error_type", 100)
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True)
class OrchestrationRequest:
    task_id: str
    run_id: str
    objective: str
    user_input: str
    mode: str
    memory_hits: tuple[Any, ...]
    evidence: tuple[Any, ...]
    allowed_tools: tuple[str, ...]
    base_messages: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        _bounded_text(self.task_id, "task_id", 120)
        _bounded_text(self.run_id, "run_id", 120)
        _bounded_text(self.objective, "objective", 2000)
        _bounded_text(self.user_input, "user_input", 20000, allow_empty=True)
        if self.mode not in {"auto", "multi"}:
            raise ValueError("orchestration request mode must be auto or multi")
        if len(self.memory_hits) > 50 or len(self.evidence) > 50:
            raise ValueError("orchestration context exceeds 50 items")
        object.__setattr__(self, "allowed_tools", _string_tuple(self.allowed_tools, "allowed_tools", item_length=120))
        if not isinstance(self.base_messages, (tuple, list)) or len(self.base_messages) > 20:
            raise ValueError("base_messages must contain at most 20 messages")
        if any(not isinstance(message, dict) for message in self.base_messages):
            raise ValueError("base_messages entries must be objects")
        object.__setattr__(self, "base_messages", tuple(dict(message) for message in self.base_messages))


@dataclass(frozen=True)
class ExecutorResult:
    answer: str
    status: str
    pending_tool_calls: list[dict[str, Any]]
    trace: dict[str, Any]
    tool_call_data: dict[str, Any] | None = None
    tool_context: list[dict[str, Any]] | None = None
    proposal_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_rounds: int = 0
    tool_warning: str | None = None
    model_calls: int | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.answer, "answer", 20000, allow_empty=True)
        if self.status not in {"succeeded", "awaiting_tool_approval", "failed"}:
            raise ValueError("unsupported executor status")
        if not isinstance(self.pending_tool_calls, list) or not all(isinstance(item, dict) for item in self.pending_tool_calls):
            raise ValueError("pending_tool_calls must be a list of objects")
        if not isinstance(self.trace, dict):
            raise ValueError("executor trace must be an object")
        if self.tool_context is not None and not isinstance(self.tool_context, list):
            raise ValueError("tool_context must be a list or None")
        if not isinstance(self.proposal_calls, list) or not all(isinstance(item, dict) for item in self.proposal_calls):
            raise ValueError("proposal_calls must be a list of objects")
        if not isinstance(self.tool_rounds, int) or isinstance(self.tool_rounds, bool) or self.tool_rounds < 0:
            raise ValueError("tool_rounds must be a non-negative integer")
        if self.tool_warning is not None:
            _bounded_text(self.tool_warning, "tool_warning", 100)
        if self.model_calls is not None and (
            not isinstance(self.model_calls, int) or isinstance(self.model_calls, bool) or self.model_calls < 0
        ):
            raise ValueError("model_calls must be a non-negative integer or None")


@dataclass(frozen=True)
class OrchestrationResult:
    answer: str
    status: str
    trace: dict[str, Any]
    selected_plan: dict[str, Any] | None = None
    pending_tool_calls: list[dict[str, Any]] | None = None
    executor_result: ExecutorResult | None = None
