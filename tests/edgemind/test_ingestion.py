import json

from edgemind.ingestion import assign_chunk_ids, build_corpus_manifest, load_chunks_json, stable_chunk_id


def _chunk(document, page, content):
    return {"content": content, "metadata": {"file_name": document, "page": page}}


def test_identical_input_produces_stable_chunk_and_corpus_ids():
    first = _chunk("annual.pdf", 1, "Revenue: 10\r\nCost: 5")
    second = _chunk("annual.pdf", 1, "Revenue: 10\nCost: 5")
    assert stable_chunk_id("annual.pdf", 1, first["content"]) == stable_chunk_id("annual.pdf", 1, second["content"])
    manifest_a = build_corpus_manifest([first], "fitz", "1.0", {"chunk_size": 800})
    manifest_b = build_corpus_manifest([second], "fitz", "1.0", {"chunk_size": 800})
    assert manifest_a["corpus_version"] == manifest_b["corpus_version"]


def test_content_change_changes_chunk_and_corpus_version():
    first = _chunk("annual.pdf", 1, "Revenue: 10")
    changed = _chunk("annual.pdf", 1, "Revenue: 11")
    assert stable_chunk_id("annual.pdf", 1, first["content"]) != stable_chunk_id("annual.pdf", 1, changed["content"])
    assert build_corpus_manifest([first], "fitz", "1.0", {})["corpus_version"] != build_corpus_manifest([changed], "fitz", "1.0", {})["corpus_version"]


def test_duplicate_pages_are_distinguished_by_document_identity():
    one = _chunk("company-a.pdf", 1, "Same page")
    two = _chunk("company-b.pdf", 1, "Same page")
    assigned = assign_chunk_ids([one, two])
    assert assigned[0]["chunk_id"] != assigned[1]["chunk_id"]


def test_legacy_chunk_json_loads_and_derives_ids_without_mutation(tmp_path):
    path = tmp_path / "legacy.json"
    original = [_chunk("annual.pdf", 2, "Legacy text")]
    encoded = json.dumps(original, ensure_ascii=False)
    path.write_text(encoded, encoding="utf-8")
    loaded = load_chunks_json(path)
    assert loaded[0]["chunk_id"]
    assert "chunk_id" not in original[0]
    assert path.read_text(encoding="utf-8") == encoded
