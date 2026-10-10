import math
import uuid
from datetime import datetime, timezone

from .models import MEMORY_LAYERS, MemoryHit, MemoryItem
from .store import HarnessStore


class MemoryService:
    def __init__(self, store: HarnessStore, embed_query=None):
        self.store = store
        self.embed_query = embed_query

    def propose(self, layer: str, content: str, source_task_id: str | None = None,
                source_run_id: str | None = None, confidence: float = 0.5,
                metadata: dict | None = None) -> MemoryItem:
        content = content.strip()
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"unknown memory layer: {layer}")
        if not content:
            raise ValueError("memory content is required")
        if source_task_id and self.store.get_task(source_task_id) is None:
            raise ValueError(f"unknown source task: {source_task_id}")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        item = MemoryItem(str(uuid.uuid4()), layer, content, "pending", source_task_id,
                          source_run_id, confidence, metadata or {}, self.make_embedding(content))
        return self.store.put_memory(item)

    def make_embedding(self, content: str) -> list[float] | None:
        if not self.embed_query:
            return None
        try:
            vector = self.embed_query(content)
            return [float(value) for value in vector] if vector else None
        except Exception:
            return None

    def review(self, memory_id: str, decision: str, edited_content: str | None = None) -> MemoryItem:
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        if edited_content is not None and not edited_content.strip():
            raise ValueError("edited memory content cannot be empty")
        item = self.store.review_memory(memory_id, decision, edited_content.strip() if edited_content else None)
        if item.source_task_id:
            self.store.append_event(item.source_task_id, "memory_reviewed", {"memory_id": memory_id, "decision": decision})
        return item

    def delete(self, memory_id: str) -> bool:
        item = self.store.get_memory(memory_id)
        if item and item.source_task_id:
            self.store.append_event(item.source_task_id, "memory_deleted", {"memory_id": memory_id})
        return self.store.delete_memory(memory_id)

    def retrieve(self, query: str, task_id: str | None = None, limit: int = 8,
                 include_pending: bool = False, layer_weights: dict[str, float] | None = None,
                 time_decay_days: float | None = None, allowed_tags: tuple[str, ...] = (),
                 excluded_tags: tuple[str, ...] = ()) -> list[MemoryHit]:
        if not query.strip() or limit <= 0:
            return []
        defaults = {"working": 1.15, "episodic": 1.0, "semantic": 1.1, "procedural": 1.0}
        weights = defaults if layer_weights is None else dict(layer_weights)
        if set(weights) != set(MEMORY_LAYERS) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2
            for value in weights.values()
        ):
            raise ValueError("layer_weights must contain valid weights for all memory layers")
        if time_decay_days is not None and (
            isinstance(time_decay_days, bool) or not isinstance(time_decay_days, (int, float))
            or not math.isfinite(time_decay_days) or time_decay_days <= 0
        ):
            raise ValueError("time_decay_days must be a positive finite number")
        if not query.strip() or limit <= 0:
            return []
        lexical = self.store.search_memories(query, limit=max(limit * 3, 20), include_pending=include_pending, task_id=task_id)
        lexical_scores = {item.memory_id: score for item, score in lexical}
        semantic_scores = {}
        semantic_items = {}
        if self.embed_query:
            try:
                query_vector = self.embed_query(query)
                if query_vector:
                    candidates = self.store.list_memories(status="approved", limit=10000)
                    semantic_items = {item.memory_id: item for item in candidates
                                      if (not task_id or not item.source_task_id or item.source_task_id == task_id) and item.embedding}
                    semantic_scores = {memory_id: self._cosine(query_vector, item.embedding)
                                       for memory_id, item in semantic_items.items()}
            except Exception:
                semantic_scores = {}
        hits = []
        by_id = {item.memory_id: item for item, _ in lexical}
        by_id.update(semantic_items)
        allowed, excluded = set(allowed_tags), set(excluded_tags)
        for memory_id in set(lexical_scores) | set(semantic_scores):
            item = by_id[memory_id]
            tags_value = item.metadata.get("tags", []) if isinstance(item.metadata, dict) else []
            tags = set(tags_value) if isinstance(tags_value, (list, tuple, set)) else set()
            if allowed and not (tags & allowed):
                continue
            if excluded & tags:
                continue
            lex_score = lexical_scores.get(memory_id, 0.0)
            sem_score = semantic_scores.get(memory_id, 0.0)
            recency = self._recency(item.updated_at, time_decay_days or 30.0)
            layer_weight = weights.get(item.layer, 1.0)
            fused = (0.55 * lex_score + 0.35 * sem_score + 0.1 * recency) * layer_weight
            hits.append(MemoryHit(item, fused, {"lexical": lex_score, "semantic": sem_score,
                                                "recency": recency, "layer_weight": layer_weight}))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
    @staticmethod
    def _cosine(left, right) -> float:
        if len(left) != len(right):
            return 0.0
        denom = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
        return 0.0 if denom == 0 else max(0.0, sum(x * y for x, y in zip(left, right)) / denom)

    @staticmethod
    def _recency(updated_at: str, decay_days: float = 30.0) -> float:
        try:
            created = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86400)
            return 1.0 / (1.0 + age_days / decay_days)
        except (TypeError, ValueError):
            return 0.0
