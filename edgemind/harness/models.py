from dataclasses import dataclass, field
from typing import Any


MEMORY_LAYERS = {"working", "episodic", "semantic", "procedural"}
MEMORY_STATUSES = {"pending", "approved", "rejected", "superseded"}
TASK_STATUSES = {"active", "paused", "completed", "failed"}


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    title: str
    objective: str
    status: str
    state: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    layer: str
    content: str
    status: str
    source_task_id: str | None
    source_run_id: str | None
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class HarnessEvent:
    event_id: str
    task_id: str
    run_id: str | None
    event_type: str
    payload: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class MemoryHit:
    item: MemoryItem
    score: float
    explanation: dict[str, float]


@dataclass(frozen=True)
class AgentTurn:
    run_id: str
    task_id: str
    status: str
    answer: str
    trace: dict[str, Any]
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
