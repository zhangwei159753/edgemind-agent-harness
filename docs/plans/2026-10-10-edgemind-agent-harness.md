# EdgeMind Agent Harness 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first personal Agent Harness with persistent tasks, four-layer memory, multimodal input, guarded MCP tools, and preserve EdgeMind's RAG evaluation plane.

**Architecture:** Add an isolated edgemind.harness package and a separate SQLite store with versioned tables. Keep orchestration behind provider, media, memory, retrieval, and MCP adapters; let the existing Gradio app consume application services without importing legacy RAG at startup.

**Tech Stack:** Python 3.11+, SQLite/FTS5, Gradio, OpenAI-compatible DeepSeek API, optional local Whisper-compatible ASR, official MCP Python SDK.

**Spec:** docs/specs/2026-10-10-edgemind-long-term-agent-harness.md

## Global Constraints

- Default host is 127.0.0.1; API credentials are read from environment and never persisted.
- Existing RAG commands, database tables, evaluation gates, and citations remain usable.
- Pending semantic/procedural memories are excluded from default retrieval.
- Side-effecting MCP tools require explicit user confirmation.
- Audio processing is optional and must fail with an actionable message when the ASR extra is absent.
- Do not claim quality improvement without a recorded benchmark result.

## Review Focus

- Existing SQLite database has legacy schema and data; Harness tables must be additive and initialization idempotent.
- DeepSeek text and image messages have different content shapes; image parts must only appear in user messages.
- Attachments with invalid extension, MIME, size, or missing files are rejected safely.
- MCP server launch/call can fail or time out; errors are bounded and secrets are never persisted.
- Memory conflicts or deletion cannot leave stale FTS/embedding records visible in retrieval.

---

### Task 1: Harness data model and persistence

**Files:** Create edgemind/harness/__init__.py, models.py, store.py; add tests under tests/edgemind/harness/test_store.py.

**Interfaces:** Immutable TaskRecord, MemoryItem, HarnessEvent, AgentTurn dataclasses. HarnessStore methods: create_task, get_task, list_tasks, update_task_state, append_event, put_memory, get_memory, list_memories, review_memory, delete_memory, search_memories, schema_version. Layers: working, episodic, semantic, procedural. Candidate states: pending, approved, rejected, superseded.

- [ ] Add behavior tests for additive schema, task restart persistence, review filtering, and deletion; observe expected failure.
- [x] Implement versioned additive schema and typed conversion routines.
- [ ] Run focused and existing storage tests.
- [ ] Commit feat: add persistent harness store (source directory has no Git metadata).

### Task 2: Memory lifecycle and retrieval

**Files:** Create edgemind/harness/memory.py and retrieval.py; modify harness/store.py; add tests/edgemind/harness/test_memory.py.

**Interfaces:** MemoryService.propose(...) -> MemoryItem; review(memory_id, decision, edited_content=None) -> MemoryItem; retrieve(query, task_id=None, limit=8, include_pending=False) -> list[MemoryHit]. MemoryHit carries item, fused score, and per-signal explanation.

- [ ] Cover candidate visibility, approved-only filtering, edits, deletion, FTS ranking, and no-embedding fallback.
- [x] Implement FTS5 keyword retrieval and optional embedding adapter; fuse available ranks deterministically with recency/layer boosts.
- [ ] Keep pending memories invisible unless requested by review panel.
- [ ] Inspect retrieval explanation output.
- [ ] Commit feat: add reviewed four-layer memory retrieval (source directory has no Git metadata).

### Task 3: Persistent task lifecycle and multimodal input

**Files:** Create harness/tasks.py and media.py; modify providers.py and pyproject.toml; add tests/edgemind/harness/test_tasks.py and test_media.py.

**Interfaces:** TaskService create, pause, resume, complete, checkpoint, context. MediaProcessor process_text, process_image, process_audio; output normalized message plus attachment metadata. Image content uses OpenAI-compatible user message parts. Audio adapter is optional and returns transcript plus source metadata.

- [ ] Cover task transitions, restart-safe checkpoints, empty input, unsupported transitions, media validation, image content encoding, and missing-ASR error.
- [x] Implement bounded attachment validation (image 20 MiB, audio 40 MiB); raw media is not stored by default.
- [x] Add optional audio and MCP dependency groups, not in default installation.
- [ ] Commit feat: persist tasks and multimodal input (source directory has no Git metadata).

### Task 4: Agent orchestration and guarded MCP

**Files:** Create harness/agent.py, tools.py, mcp_client.py; modify providers.py and pyproject.toml; add test_agent.py and test_tools.py.

**Interfaces:** AgentService.run_turn(task_id, text, attachments=()) -> AgentTurn. ToolRegistry list_enabled, propose_call, approve_call, execute_call. McpClientManager list_tools(server_name), call_tool(server_name, tool_name, arguments). A turn has bounded tool rounds/timeouts and emits an event per state transition.

- [ ] Cover text/image messages, memory context injection, user correction to pending semantic memory, failure checkpointing, bounded tool rounds, approval before side effects, and redaction.
- [x] Implement DeepSeek function-call proposal flow with injectable API client and explicit confirmation.
- [x] Add MCP stdio lifecycle through the optional official SDK; validate server config and use environment references for secrets.
- [x] Provide no shell/code execution builtin.
- [ ] Commit feat: add guarded agent and MCP runtime (source directory has no Git metadata).

### Task 5: Unified UI and documentation

**Files:** Modify edgemind/web.py, README.md, .env.example; create docs/mcp-servers.example.json; extend tests/edgemind/test_web.py.

- [ ] Verify app startup does not import myrag; UI handlers do not return API keys.
- [x] Add GPT-style light UI with persistent left navigation/task selector, centered chat, bottom composer, Agent workspace, task resume, memory review, attachment input, tool confirmation, and run trace while preserving API/RAG diagnostics.
- [ ] Keep local-only binding and make optional dependency errors actionable.
- [x] Document dependencies, DeepSeek image use, local ASR setup, MCP configuration, persistence/privacy, and RAG workflow.
- [ ] Run project checks and local smoke workflow; report environment limitations accurately.
- [ ] Commit feat: ship EdgeMind Agent Harness workspace (source directory has no Git metadata).

## Delivery Notes

- Implemented in the source tree: additive Harness package and SQLite schema, task lifecycle/checkpoints, four-layer reviewed memory, image input, optional local ASR, MCP stdio adapter with approval, lazy RAG integration, correction flow, CLI memory export, GPT-style Gradio workspace, and setup documentation.
- Automated tests and live provider/MCP/ASR workflows were not run in this delivery. The code still needs environment-backed integration verification before claiming end-to-end reliability.
- The source directory has no Git metadata, so this delivery does not include commits or branch history.
- Conflict detection/deduplication and automatic promotion of Harness prompt/retrieval settings remain follow-up work; they are not represented as shipped self-evolution features.
