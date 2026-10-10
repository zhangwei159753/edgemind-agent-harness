import time
import uuid
from datetime import datetime, timezone

from .ingestion import assign_chunk_ids
from .models import FailureAttribution, QueryRun, RetrievalConfig


class QueryService:
    def __init__(self, rag_factory, store, corpus_version: str, config_version: str):
        self.rag_factory = rag_factory
        self.store = store
        self.corpus_version = corpus_version
        self.config_version = config_version

    def ask(self, question: str, top_k: int = 5, config: RetrievalConfig | None = None) -> QueryRun:
        run_id = str(uuid.uuid4())
        started = time.perf_counter()
        stage_ms: dict[str, float] = {}
        evidence: list[dict] = []
        citations: list[dict] = []
        answer = ""
        abstained = None
        status = "succeeded"
        error_type = None
        try:
            rag = self.rag_factory(config)
            result = rag.generate_answer(question, top_k=top_k)
            elapsed = (time.perf_counter() - started) * 1000
            raw_evidence = list(result.get("retrieval_chunks") or result.get("evidence") or [])
            evidence = assign_chunk_ids(raw_evidence)
            citations = self._map_citations(self._citations(result), evidence)
            answer = str(result.get("answer") or "")
            abstained = result.get("abstained")
            stage_ms.update({str(key): float(value) for key, value in (result.get("stage_ms") or {}).items()})
            stage_ms.setdefault("total", round(elapsed, 3))
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            stage_ms["total"] = round((time.perf_counter() - started) * 1000, 3)
        run = QueryRun(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            question=question,
            answer=answer,
            evidence=evidence,
            corpus_version=self.corpus_version,
            config_version=self.config_version,
            stage_ms=stage_ms,
            status=status,
            abstained=abstained,
            error_type=error_type,
            citations=citations,
        )
        self.store.record_run(run)
        attributions = self.diagnose(run)
        if attributions:
            self.store.record_attributions(run_id, attributions)
        return run

    @staticmethod
    def _citations(result: dict) -> list[dict]:
        if "citations" in result:
            return list(result.get("citations") or [])
        if "citation_ids" in result:
            return [{"chunk_id": str(chunk_id)} for chunk_id in (result.get("citation_ids") or [])]
        if result.get("filename") or result.get("page"):
            return [{"filename": result.get("filename", ""), "page": result.get("page", "")}]
        return []

    @staticmethod
    def _map_citations(citations: list[dict], evidence: list[dict]) -> list[dict]:
        mapped = []
        for citation in citations:
            item = dict(citation)
            if item.get("chunk_id"):
                match = next((chunk for chunk in evidence if chunk.get("chunk_id") == str(item["chunk_id"])), None)
            else:
                filename = str(item.get("filename") or item.get("file_name") or item.get("source") or "")
                page = str(item.get("page", ""))
                matches = []
                for chunk in evidence:
                    metadata = chunk.get("metadata") or {}
                    chunk_filename = str(chunk.get("filename") or metadata.get("file_name") or metadata.get("filename") or metadata.get("source") or "")
                    chunk_page = str(chunk.get("page", metadata.get("page", "")))
                    if filename and filename == chunk_filename and page == chunk_page:
                        matches.append(chunk)
                    elif not filename and page and page == chunk_page:
                        matches.append(chunk)
                match = matches[0] if len(matches) == 1 else None
            if match:
                item["chunk_id"] = match["chunk_id"]
            mapped.append(item)
        return mapped

    @staticmethod
    def diagnose(run: QueryRun) -> list[FailureAttribution]:
        items = []
        if run.status == "failed":
            category, reason, label = "query_failure", f"Query failed with {run.error_type or 'unknown error'}", "ungrounded_answer"
        elif not run.evidence:
            category, reason, label = "empty_retrieval", "No evidence chunks were retrieved", "missing_document"
        elif not run.citations:
            category, reason, label = "missing_citation", "Evidence was retrieved but no final citation was returned", "wrong_citation"
        else:
            return items
        items.append(FailureAttribution(str(uuid.uuid4()), run.run_id, category, reason, label, False, None))
        return items
