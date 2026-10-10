import json
import time
from pathlib import Path
from statistics import mean
from typing import Iterable


def recall_at_k(retrieved_ids: Iterable[str], expected_ids: Iterable[str], k: int = 5) -> float | None:
    expected = set(expected_ids)
    if not expected:
        return None
    return len(set(list(retrieved_ids)[:k]) & expected) / len(expected)


def mean_reciprocal_rank(retrieved_ids: Iterable[str], expected_ids: Iterable[str]) -> float | None:
    expected = set(expected_ids)
    if not expected:
        return None
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def _macro(values: list[float]) -> float | None:
    return mean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return round(ordered[index], 3)


class EvaluationRunner:
    def __init__(self, query_service_factory):
        self.query_service_factory = query_service_factory

    def run(self, dataset_path, config=None, config_version: str = "baseline", top_k: int = 5) -> dict:
        samples = []
        durations = []
        dataset = Path(dataset_path)
        for line_number, raw in enumerate(dataset.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
                if not isinstance(item, dict) or not item.get("example_id") or not item.get("question"):
                    raise ValueError("each sample requires example_id and question")
                expected = item.get("expected_chunk_ids")
                if expected is not None and (not isinstance(expected, list) or not all(isinstance(value, str) for value in expected)):
                    raise ValueError("expected_chunk_ids must be a list of strings or null")
                if item.get("answerable") is not None and not isinstance(item["answerable"], bool):
                    raise ValueError("answerable must be a boolean or null")
            except Exception as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc

            started = time.perf_counter()
            row = {
                "example_id": item["example_id"],
                "question": item["question"],
                "expected_chunk_ids": expected,
                "answerable": item.get("answerable"),
                "retrieved_ids": [],
                "citation_ids": [],
                "abstained": None,
                "status": "failed",
                "error_type": None,
                "elapsed_ms": None,
            }
            try:
                query_service = self.query_service_factory(config, config_version)
                run = query_service.ask(item["question"], top_k=top_k)
                row["retrieved_ids"] = [self._chunk_id(chunk) for chunk in run.evidence if self._chunk_id(chunk)]
                row["citation_ids"] = [self._citation_id(citation) for citation in run.citations if self._citation_id(citation)]
                row["abstained"] = run.abstained
                row["status"] = run.status
                row["error_type"] = run.error_type
            except Exception as exc:
                row["error_type"] = type(exc).__name__
            elapsed = (time.perf_counter() - started) * 1000
            row["elapsed_ms"] = round(elapsed, 3)
            durations.append(elapsed)
            samples.append(row)
        return {
            "config_version": config_version,
            "top_k": top_k,
            "sample_count": len(samples),
            "failure_count": sum(row["status"] == "failed" for row in samples),
            "samples": samples,
            "metrics": self._metrics(samples, durations, top_k),
        }

    @staticmethod
    def _chunk_id(chunk: dict) -> str | None:
        value = chunk.get("chunk_id") or chunk.get("id")
        return None if value is None else str(value)

    @staticmethod
    def _citation_id(citation: dict | str) -> str | None:
        if isinstance(citation, str):
            return citation
        value = citation.get("chunk_id") or citation.get("id")
        return None if value is None else str(value)

    @staticmethod
    def _metrics(samples: list[dict], durations: list[float], top_k: int) -> dict:
        labeled = [row for row in samples if row["expected_chunk_ids"]]
        recalls = [recall_at_k(row["retrieved_ids"], row["expected_chunk_ids"], top_k) for row in labeled]
        reciprocal_ranks = [mean_reciprocal_rank(row["retrieved_ids"], row["expected_chunk_ids"]) for row in labeled]
        citation_hits = [
            1.0 if set(row["citation_ids"]) & set(row["expected_chunk_ids"]) else 0.0
            for row in labeled
        ]
        labeled_answerability = [row for row in samples if isinstance(row["answerable"], bool) and isinstance(row["abstained"], bool)]
        abstention_scores = [float(row["abstained"] == (not row["answerable"])) for row in labeled_answerability]
        return {
            "recall_at_k": _macro(recalls),
            "mrr": _macro(reciprocal_ranks),
            "citation_hit_rate": _macro(citation_hits),
            "no_answer_accuracy": _macro(abstention_scores),
            "latency_ms": {
                "count": len(durations),
                "mean": round(mean(durations), 3) if durations else None,
                "p95": _percentile(durations, 0.95),
            },
            "cost": None,
        }
