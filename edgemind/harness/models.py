from dataclasses import dataclass, field
import math
from typing import Any


MEMORY_LAYERS = {"working", "episodic", "semantic", "procedural"}
MEMORY_STATUSES = {"pending", "approved", "rejected", "superseded"}
TASK_STATUSES = {"active", "paused", "completed", "failed"}
MEMORY_RELATIONS = {"derived_from", "supports", "contradicts", "used_in", "similar_to"}
MEMORY_EDGE_STATUSES = {"pending", "approved", "rejected"}


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
class MemoryEdge:
    edge_id: str
    source_memory_id: str
    target_memory_id: str
    relation: str
    status: str
    confidence: float
    source_run_id: str | None
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        for field_name in ("edge_id", "source_memory_id", "target_memory_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise ValueError(f"{field_name} must be non-empty text no longer than 200 characters")
        if self.source_memory_id == self.target_memory_id:
            raise ValueError("memory edge cannot point to itself")
        if self.relation not in MEMORY_RELATIONS:
            raise ValueError("unsupported memory edge relation")
        if self.status not in MEMORY_EDGE_STATUSES:
            raise ValueError("unsupported memory edge status")
        if (not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool)
                or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("memory edge confidence must be a finite number from 0 to 1")
        if self.source_run_id is not None and (not isinstance(self.source_run_id, str) or len(self.source_run_id) > 200):
            raise ValueError("source_run_id must be text no longer than 200 characters")
        if not isinstance(self.created_at, str) or not isinstance(self.updated_at, str):
            raise ValueError("memory edge timestamps must be text")


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
