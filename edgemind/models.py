from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_vector: int = 10
    top_k_bm25: int = 10
    fusion_weight_vector: float = 0.5
    fusion_weight_bm25: float = 0.5
    rerank_top_k: int = 5
    score_threshold: float = 0.0
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass(frozen=True)
class QueryRun:
    run_id: str
    created_at: str
    question: str
    answer: str
    evidence: list[dict[str, Any]]
    corpus_version: str
    config_version: str
    stage_ms: dict[str, float]
    status: str
    abstained: bool | None = None
    error_type: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Feedback:
    feedback_id: str
    run_id: str
    label: str
    note: str
    created_at: str


@dataclass(frozen=True)
class FailureAttribution:
    attribution_id: str
    run_id: str
    category: str
    reason: str
    suggested_label: str | None
    confirmed: bool = False
    reviewer_label: str | None = None
