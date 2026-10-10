import copy
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

ID_ALGORITHM = "sha256-v1"


def _normalize_content(content: Any) -> str:
    text = unicodedata.normalize("NFC", str(content or ""))
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _document_id(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    return str(
        chunk.get("document_id")
        or metadata.get("document_id")
        or metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("source")
        or "unknown-document"
    )


def _page(chunk: dict) -> Any:
    metadata = chunk.get("metadata") or {}
    return chunk.get("page", metadata.get("page", ""))


def stable_chunk_id(document_id: str, page: Any, content: str) -> str:
    payload = {
        "algorithm": ID_ALGORITHM,
        "document_id": unicodedata.normalize("NFC", str(document_id)),
        "page": str(page),
        "content": _normalize_content(content),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "chk_" + hashlib.sha256(encoded).hexdigest()


def assign_chunk_ids(chunks: list[dict]) -> list[dict]:
    assigned = []
    for chunk in chunks:
        item = copy.deepcopy(chunk)
        metadata = item.get("metadata") or {}
        item["chunk_id"] = str(item.get("chunk_id") or stable_chunk_id(_document_id(item), _page(item), item.get("content", "")))
        assigned.append(item)
    return assigned


def load_chunks_json(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as source:
        chunks = json.load(source)
    if not isinstance(chunks, list):
        raise ValueError("Chunk JSON must contain a top-level list")
    if any(not isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("Every chunk must be a JSON object")
    return assign_chunk_ids(chunks)


def build_corpus_manifest(
    chunks: list[dict], parser_name: str, parser_version: str, chunk_config: dict
) -> dict:
    entries = []
    for chunk in assign_chunk_ids(chunks):
        content_hash = hashlib.sha256(_normalize_content(chunk.get("content", "")).encode("utf-8")).hexdigest()
        entries.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": _document_id(chunk),
            "page": str(_page(chunk)),
            "content_sha256": content_hash,
        })
    entries.sort(key=lambda entry: (entry["document_id"], entry["page"], entry["chunk_id"]))
    descriptor = {
        "manifest_schema": 1,
        "id_algorithm": ID_ALGORITHM,
        "parser": {"name": str(parser_name), "version": str(parser_version)},
        "chunk_config": chunk_config,
        "chunks": entries,
    }
    encoded = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**descriptor, "corpus_version": "corpus_" + hashlib.sha256(encoded).hexdigest()}
