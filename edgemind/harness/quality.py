"""Validated, append-only feedback for Agent runs and exposed memory."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from .store import HarnessStore

OUTCOMES = {"success", "partial", "failure", "unrated"}
FAILURE_CATEGORIES = {
    "planning", "memory", "retrieval/evidence", "tool selection", "tool execution",
    "answer quality", "provider/network", "user input", "other",
}
MEMORY_RATINGS = {"helpful", "not_helpful", "uncertain"}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bauthorization\s*[:=]\s*(?:basic|bearer)\s+[A-Za-z0-9+/=_-]{8,}"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _contains_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


class QualityService:
    """Records user-provided outcome and memory utility labels for an Agent run."""

    def __init__(self, store: HarnessStore):
        self.store = store

    def record_feedback(
        self,
        task_id: str,
        run_id: str,
        outcome: str,
        failure_category: str | None = None,
        memory_feedback: dict[str, str] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        if not isinstance(outcome, str) or outcome not in OUTCOMES:
            raise ValueError("outcome must be success, partial, failure or unrated")
        if failure_category is not None and (not isinstance(failure_category, str) or failure_category not in FAILURE_CATEGORIES):
            raise ValueError("unsupported failure category")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("feedback note must be text no longer than 2,000 characters")
        if _contains_secret(note):
            raise ValueError("feedback note appears to contain a credential")
        if memory_feedback is None:
            memory_feedback = {}
        if not isinstance(memory_feedback, dict) or len(memory_feedback) > 100:
            raise ValueError("memory feedback must be a mapping of at most 100 memory IDs")
        for memory_id, rating in memory_feedback.items():
            if not isinstance(memory_id, str) or not memory_id or not isinstance(rating, str) or rating not in MEMORY_RATINGS:
                raise ValueError("memory feedback contains an invalid ID or rating")

        event = self.store.find_agent_turn_event(run_id)
        if event is None:
            raise ValueError("feedback must reference an existing agent_turn run")
        if event.task_id != task_id:
            raise ValueError("run does not belong to the supplied task")
        trace = event.payload.get("trace", {})
        hits = trace.get("memory_hits", []) if isinstance(trace, dict) else []
        exposed_ids = {
            item.get("memory_id") for item in hits
            if isinstance(item, dict) and isinstance(item.get("memory_id"), str)
        }
        if not set(memory_feedback).issubset(exposed_ids):
            raise ValueError("memory feedback must reference memory exposed in this run")

        timestamp = _now()
        payload = {
            "feedback_id": str(uuid.uuid4()),
            "task_id": task_id,
            "run_id": run_id,
            "source_event_id": event.event_id,
            "outcome": outcome,
            "failure_category": failure_category,
            "note": note.strip(),
            "created_at": timestamp,
            "memory_feedback": [
                {"memory_feedback_id": str(uuid.uuid4()), "memory_id": memory_id, "rating": rating}
                for memory_id, rating in sorted(memory_feedback.items())
            ],
        }
        self.store.insert_agent_feedback(payload)
        return {
            "feedback_id": payload["feedback_id"], "task_id": task_id, "run_id": run_id,
            "source_event_id": event.event_id, "outcome": outcome,
            "failure_category": failure_category, "note": payload["note"], "created_at": timestamp,
            "memory_feedback": dict(memory_feedback),
        }

    def list_feedback(self, task_id: str | None = None, run_id: str | None = None,
                      limit: int = 100) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        return self.store.list_agent_feedback(task_id=task_id, run_id=run_id, limit=limit)
