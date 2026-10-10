# EdgeMind Memory Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provenance-aware, human-governed relations between existing memory items and use bounded graph expansion to improve retrieval explanations.

**Architecture:** Keep four existing memory layers and `MemoryService.retrieve()` compatibility. Add SQLite relation edges whose endpoints are existing `MemoryItem` IDs, migrate schema additively, require review for model-proposed edges, and expand only approved relations by at most one hop.

**Tech Stack:** Python 3.11+, SQLite foreign keys, existing FTS/BM25 and optional embedding retrieval, Gradio, pytest; no graph database dependency.

**Spec:** `docs/specs/2026-10-10-edgemind-multi-agent-memory-graph-tot.md`

## Global Constraints

- 记忆图以现有四层记忆为节点，SQLite 关系表为边；不替换现有记忆生命周期，也不首期引入图数据库。
- 记忆边审核状态沿用 pending/approved/rejected；待审核边不得参与普通检索。
- 查询最多扩展 1 跳并设置节点数上限；可关闭图扩展回退到现有检索。
- API key、原始媒体字节和不受控网页内容不得进入关系边或轨迹。
- 现有 SQLite 状态须通过显式向前迁移升级，原始任务、记忆和反馈不可损坏。

## Review Focus

- Existing schema v1-v4 databases: Task 1 tests sequential migration to v5 and idempotent reopen.
- Deleted/rejected memory nodes: Task 1 tests cascade/no dangling edges and retained audit consistency.
- Duplicate, self, reverse and symmetric relations: Task 1 tests uniqueness and `similar_to` canonicalization.
- Pending/contradictory edges: Task 2 tests they do not silently influence normal retrieval.
- Large/high-degree graph: Task 2 tests one-hop and candidate caps preserve bounded latency/candidate count.

---

### Task 1: Edge model and additive SQLite migration

**Files:**
- Modify: `edgemind/harness/models.py`
- Modify: `edgemind/harness/store.py`
- Modify: `tests/edgemind/test_harness.py`
- Modify: `tests/edgemind/test_quality.py`
- Modify: `tests/edgemind/test_agent_policies.py`
- Modify: `tests/edgemind/test_agent_experiments.py`
- Create: `tests/edgemind/test_memory_graph_store.py`

**Interfaces:**
- Add immutable `MemoryEdge(edge_id: str, source_memory_id: str, target_memory_id: str, relation: str, status: str, confidence: float, source_run_id: str | None, created_at: str, updated_at: str)`.
- Bump `HarnessStore.schema_version` to 5 with `harness_memory_edges`; endpoints reference `harness_memories(memory_id) ON DELETE CASCADE`; check relation in `derived_from`, `supports`, `contradicts`, `used_in`, `similar_to`, status in `pending`, `approved`, `rejected`, confidence 0..1, and no self-edge.
- Store API: `put_memory_edge(edge) -> MemoryEdge`, `get_memory_edge(edge_id) -> MemoryEdge | None`, `list_memory_edges(memory_id, status="approved", limit=100) -> list[MemoryEdge]`, `review_memory_edge(edge_id, status) -> MemoryEdge`, `delete_memory_edges_for_memory(memory_id) -> int`.
- Canonicalize endpoint order for `similar_to` and enforce unique `(source_memory_id,target_memory_id,relation)`.

- [ ] **Step 1: Write failing tests** for v4→v5 migration, reopen idempotency, invalid relation/status/confidence/self-edge, duplicate insert, similar-to reverse duplicate, FK cascade.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_memory_graph_store.py tests/edgemind/test_harness.py -q`; confirm expected failures.
- [ ] **Step 3: Implement** dataclass and transactional additive migration/methods; update schema-version assertions to 5 only where they represent current schema.
- [ ] **Step 4: Run the tests**; verify existing task/memory data survives a migration fixture.
- [ ] **Step 5: Commit** as `feat: persist governed memory graph edges`.

### Task 2: Edge governance and bounded retrieval expansion

**Files:**
- Modify: `edgemind/harness/memory.py`
- Modify: `edgemind/harness/__init__.py` (public graph/retrieval result exports)
- Create: `edgemind/harness/graph.py`
- Test: `tests/edgemind/test_harness.py`
- Create: `tests/edgemind/test_memory_graph_retrieval.py`

**Interfaces:**
- `MemoryGraphService(store).propose(source_memory_id, target_memory_id, relation, confidence, source_run_id=None) -> MemoryEdge` always creates pending edges unless explicitly called by a trusted user-review path.
- `MemoryGraphService.review(edge_id, decision) -> MemoryEdge`, with decision restricted to approved/rejected.
- `MemoryGraphService.expand(seed_hits: list[MemoryHit], max_nodes: int = 12) -> GraphExpansionResult` follows only approved edges one hop and returns `hits: list[MemoryHit]` plus `paths: list[dict]`; keep relation strings out of the numeric `MemoryHit.explanation` map.
- Extend `MemoryService.retrieve(..., graph_expansion: bool = False, graph_max_nodes: int = 12)`; default remains compatible and graph is opt-in until rollout enables it.
- `AgentService` opts into bounded expansion for normal turns by default (`memory_graph_expansion=True`); callers can disable it. Expansion preserves current task scope and allowed/excluded memory-tag filters.
- Add `MemoryService.retrieve_with_trace(query: str, ..., graph_expansion: bool = False, graph_max_nodes: int = 12) -> MemoryRetrievalResult(hits: list[MemoryHit], graph_paths: list[dict])`; existing `retrieve()` returns only `hits` by delegating to this method.
- Expanded-hit score is `seed.score * edge.confidence * relation_weight`, deduplicated by maximum score; relation weights are `supports=0.9`, `derived_from=0.8`, `used_in=0.7`, `similar_to=0.6`. `contradicts` never adds a positive hit; it returns a conflict path attached to the trace for review.

- [ ] **Step 1: Write failing tests** for pending edge exclusion, approved one-hop expansion, confidence/relation weighting, edge explanation, disabled-path identity, max-node cap, and missing/deleted endpoints.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_memory_graph_retrieval.py -q`; confirm failures.
- [ ] **Step 3: Implement** graph service and bounded seed expansion; do not call an LLM during retrieval or create edges implicitly from generated content. Edge endpoints are memory IDs only; task/evidence/run identifiers remain provenance metadata and do not become graph nodes in this release.
- [ ] **Step 4: Run tests** and verify existing `MemoryService.retrieve()` callers return equivalent results with default arguments.
- [ ] **Step 5: Commit** as `feat: add governed graph memory retrieval`.

### Task 3: Review UI, trace, and graph fixtures

**Files:**
- Modify: `edgemind/web.py`
- Modify: `edgemind/harness/agent.py`
- Create: `tests/edgemind/test_memory_graph_ui.py`
- Modify: `tests/edgemind/test_web.py`
- Create: `tests/data/memory_graph_v1.jsonl` or add fixture graph to the isolated Agent dataset.

**Interfaces:**
- Review UI provides list/pending status, source/target IDs, relation, confidence and approve/reject actions; unresolved IDs fail closed.
- Agent trace adds `memory_graph_paths: list[dict]` with IDs, relation and score components only (bounded list), never full hidden reasoning.

- [ ] **Step 1: Write failing callback tests** for review decisions, rejected-edge exclusion and concise path rendering.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_memory_graph_ui.py tests/edgemind/test_web.py -q`; confirm failures.
- [ ] **Step 3: Add** review controls and graph trace output using existing memory review patterns.
- [ ] **Step 4: Add fixture cases** where a connected approved memory supplies needed context and a pending edge is ignored; assert fixture mode never writes production DB.
- [ ] **Step 5: Commit** as `feat: expose memory graph review and trace`.

### Task 4: Migration and retrieval regression

**Files:**
- Modify: `README.md`
- Test: `tests/edgemind/test_memory_graph_store.py`
- Test: `tests/edgemind/test_memory_graph_retrieval.py`
- Test: all `tests/edgemind`

- [ ] **Step 1: Add regression** for current `.data/edgemind.sqlite3`-style schema v4 fixture with pre-existing memories and feedback; assert no data loss.
- [ ] **Step 2: Document** relation types, approval, retrieval cap, disabling behavior, and migration.
- [ ] **Step 3: Run** `python -m pytest tests/edgemind -q`; expect all tests pass.
- [ ] **Step 4: Commit** as `docs: document memory graph behavior`.
