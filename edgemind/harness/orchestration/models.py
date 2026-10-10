from dataclasses import dataclass
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
