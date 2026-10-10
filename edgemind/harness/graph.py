from __future__ import annotations

import uuid
from dataclasses import dataclass

from .models import MemoryEdge, MemoryHit
from .store import HarnessStore


RELATION_WEIGHTS = {
    "supports": 0.9,
    "derived_from": 0.8,
    "used_in": 0.7,
    "similar_to": 0.6,
}
_MAX_GRAPH_NODES = 50


@dataclass(frozen=True)
class GraphExpansionResult:
    hits: list[MemoryHit]
    paths: list[dict]


@dataclass(frozen=True)
class MemoryRetrievalResult:
    hits: list[MemoryHit]
    graph_paths: list[dict]


class MemoryGraphService:
    """Human-governed relations between approved memory nodes."""

    def __init__(self, store: HarnessStore):
        self.store = store

    def propose(self, source_memory_id: str, target_memory_id: str, relation: str,
                confidence: float, source_run_id: str | None = None) -> MemoryEdge:
        if self.store.get_memory(source_memory_id) is None or self.store.get_memory(target_memory_id) is None:
            raise ValueError("memory edge endpoints must exist")
        edge = MemoryEdge(str(uuid.uuid4()), source_memory_id, target_memory_id, relation,
                           "pending", confidence, source_run_id, "", "")
        return self.store.put_memory_edge(edge)

    def review(self, edge_id: str, decision: str) -> MemoryEdge:
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        edge = self.store.get_memory_edge(edge_id)
        if edge is None:
            raise ValueError(f"unknown memory edge: {edge_id}")
        if decision == "approved":
            endpoints = (self.store.get_memory(edge.source_memory_id),
                         self.store.get_memory(edge.target_memory_id))
            if any(item is None or item.status != "approved" for item in endpoints):
                raise ValueError("both memory edge endpoints must be approved before edge approval")
        return self.store.review_memory_edge(edge_id, decision)

    def expand(self, seed_hits: list[MemoryHit], max_nodes: int = 12) -> GraphExpansionResult:
        if (not isinstance(max_nodes, int) or isinstance(max_nodes, bool)
                or not 1 <= max_nodes <= _MAX_GRAPH_NODES):
            raise ValueError(f"max_nodes must be an integer from 1 to {_MAX_GRAPH_NODES}")
        if not isinstance(seed_hits, list):
            raise ValueError("seed_hits must be a list")

        hits_by_id: dict[str, MemoryHit] = {}
        ordered_seeds = sorted(seed_hits, key=lambda hit: hit.score, reverse=True)
        for hit in ordered_seeds:
            if hit.item.status != "approved":
                continue
            existing = hits_by_id.get(hit.item.memory_id)
            if existing is None or hit.score > existing.score:
                hits_by_id[hit.item.memory_id] = hit
            if len(hits_by_id) >= max_nodes:
                break
        seed_ids = tuple(hits_by_id)
        paths: list[dict] = []
        expanded_scores: dict[str, MemoryHit] = {}
        for seed_id in seed_ids:
            seed = hits_by_id[seed_id]
            edge_limit = min(100, max_nodes * 4)
            edges = sorted(
                self.store.list_memory_edges(seed_id, status="approved", limit=edge_limit),
                key=lambda edge: (edge.relation, edge.target_memory_id if edge.source_memory_id == seed_id
                                  else edge.source_memory_id, edge.edge_id),
            )
            for edge in edges:
                related_id = (edge.target_memory_id if edge.source_memory_id == seed_id
                              else edge.source_memory_id)
                if related_id in seed_ids:
                    continue
                relation_weight = RELATION_WEIGHTS.get(edge.relation, 0.0)
                score = seed.score * edge.confidence * relation_weight
                path = {
                    "seed_memory_id": seed_id,
                    "source_memory_id": edge.source_memory_id,
                    "target_memory_id": edge.target_memory_id,
                    "related_memory_id": related_id,
                    "relation": edge.relation,
                    "confidence": edge.confidence,
                    "relation_weight": relation_weight,
                    "score": score,
                    "kind": "conflict" if edge.relation == "contradicts" else "related",
                }
                if len(paths) < max_nodes:
                    paths.append(path)
                if edge.relation == "contradicts":
                    continue
                item = self.store.get_memory(related_id)
                if item is None or item.status != "approved":
                    continue
                candidate = MemoryHit(item, score, {
                    "graph_seed_score": seed.score,
                    "graph_edge_confidence": edge.confidence,
                    "graph_relation_weight": relation_weight,
                })
                previous = expanded_scores.get(related_id)
                if previous is None or candidate.score > previous.score:
                    expanded_scores[related_id] = candidate
        room = max(0, max_nodes - len(hits_by_id))
        best_expansions = sorted(expanded_scores.values(), key=lambda hit: hit.score, reverse=True)[:room]
        merged = list(hits_by_id.values()) + best_expansions
        merged.sort(key=lambda hit: hit.score, reverse=True)
        return GraphExpansionResult(merged[:max_nodes], paths[:max_nodes])
