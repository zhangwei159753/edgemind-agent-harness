# EdgeMind Agent Quality & Experience Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for Native execution. Steps use checkbox syntax for tracking.

**Goal:** Build a replayable, measurable and human-governed improvement loop for EdgeMind Agent behavior and memory utility.

**Architecture:** Keep the existing Harness and RAG evaluation separate. Add additive SQLite persistence for run feedback and versioned Agent policies; inject the active policy into AgentService; build a fixture-first Agent evaluation runner and a candidate comparison/gating service; expose feedback, reports, promotion and rollback in Gradio. Skills and MCP permissions remain upper bounds and cannot be widened by a policy.

**Tech Stack:** Python 3.11+, SQLite, existing Gradio 6 UI, pytest, JSONL datasets, existing DeepSeek-compatible provider adapter.

**Spec:** docs/specs/2026-10-10-edgemind-agent-quality-experience-loop.md

## Global Constraints

- Work in the existing EdgeMind tree; the user previously selected Native execution.
- Preserve existing task, RAG, memory, media, Skills, built-in tool and MCP approval behavior.
- SQLite changes are additive, explicitly versioned, and preserve existing records.
- Evaluation defaults to an isolated temporary HarnessStore, deterministic provider stub and fixture tools; it must not change the user's production task or memory state.
- Live provider evaluation is only initiated by an explicit user action; mock/fixture results and live results are visibly distinct.
- Never persist API keys, Authorization headers, raw image/audio bytes, or unrestricted external page content in runs, feedback, policies, datasets or reports.
- Policy fields are JSON data only; reject code fields, unregistered tools, invalid bounds and secret-like values.
- Skills can only narrow policy tool access; MCP side-effect confirmation remains unchanged.
- Metrics with missing labels, gold data or provider usage are N/A, not zero.
- User approval is required to promote a policy; safety gate failure always blocks promotion.
- EdgeMind is not a Git repository; do not add impossible commit steps or claim a Git diff.

## Review Focus

1. **Old schema v1 database:** migrate additively and preserve task, event, memory and MCP approval rows. Test in Task 1.
2. **Feedback for an unknown run or memory not present in that run:** reject it without inserting partial feedback. Test in Task 1.
3. **Malformed policy, dynamic-code field, secret-looking value or unknown tool:** reject candidate creation and keep active policy unchanged. Test in Task 2.
4. **Skills/MCP permission intersection:** a policy cannot reveal tools hidden by a selected Skill or bypass MCP confirmation. Test in Task 3.
5. **Missing labels, unsafe tool attempt, missing baseline metric or evaluator failure:** report N/A/block promotion without switching active policy. Test in Tasks 4–5.

---

### Task 1: Run Outcome and Memory Utility Feedback

**Files:**
- Modify: edgemind/harness/store.py
- Create: edgemind/harness/quality.py
- Test: tests/edgemind/test_quality.py
- Modify: tests/edgemind/test_harness.py

**Interfaces:**
- Produces QualityService(store: HarnessStore).
- QualityService.record_feedback(task_id: str, run_id: str, outcome: str, failure_category: str | None = None, memory_feedback: dict[str, str] | None = None, note: str = "") -> dict.
- outcome is success, partial, failure or unrated; memory feedback values are helpful, not_helpful or uncertain.
- QualityService.list_feedback(task_id: str | None = None, run_id: str | None = None, limit: int = 100) -> list[dict].
- Feedback must reference an existing agent_turn event; memory IDs must be a subset of that event's trace.memory_hits. Each feedback submission is append-only.

- [ ] **Step 1: Write migration and feedback tests**
  - Add a v1 database fixture containing a task, event, memory and pending MCP approval; reopening must preserve all four.
  - Assert valid run and memory feedback can be recorded and read back with IDs/timestamps.
  - Assert unknown run, cross-task run, unexposed memory ID, invalid outcome/category/value, oversized note and secret-like content are rejected without partial rows.

- [ ] **Step 2: Run tests to verify failure**
  - Run: .venv\Scripts\python.exe -m pytest tests/edgemind/test_quality.py tests/edgemind/test_harness.py -q
  - Expected: FAIL because feedback persistence and migration v2 are absent.

- [ ] **Step 3: Add schema-v2 feedback persistence**
  - Add append-only harness_agent_feedback and harness_agent_memory_feedback tables and indexes through an explicit v1→v2 migration in HarnessStore.
  - Expose store methods to insert and list feedback transactionally; do not rewrite existing event payloads or memory rows.

- [ ] **Step 4: Implement QualityService validation**
  - Validate the source agent_turn, task ownership, exposed memory IDs, enum values and bounded/redacted note before opening the insert transaction.
  - Store feedback only after all validation succeeds.

- [ ] **Step 5: Run focused persistence tests**
  - Run the command from Step 2.
  - Expected: PASS; legacy task/memory/tool tables remain intact and schema_version is 2 after Task 1; Task 2 advances it to 3 and Task 5 advances it to 4.

---

### Task 2: Versioned Agent Policy Repository

**Files:**
- Modify: edgemind/harness/store.py
- Create: edgemind/harness/policies.py
- Test: tests/edgemind/test_agent_policies.py

**Interfaces:**
- Produces frozen AgentPolicy(policy_id: str, version: int, prompt_suffix: str, prompt_template_ref: str | None, memory_top_k: int, layer_weights: Mapping[str, float], time_decay_days: float | None, allowed_memory_tags: tuple[str, ...], excluded_memory_tags: tuple[str, ...], tool_allowlist: tuple[str, ...], max_tool_rounds: int, output_constraints: Mapping[str, object], checker_ids: tuple[str, ...], parent_policy_id: str | None).
- Produces AgentPolicyService(store: HarnessStore) with get_active() -> AgentPolicy, get(policy_id: str) -> AgentPolicy | None, list_policies() -> list[dict], create_candidate(payload: dict, rationale: str, source_run_id: str | None = None) -> AgentPolicy, promote(policy_id: str, experiment_id: str, confirmed: bool) -> AgentPolicy, and rollback(policy_id: str, confirmed: bool) -> AgentPolicy.
- Default policy is an immutable built-in baseline used when no active record exists.
- Candidate configuration may narrow currently registered tool availability only; it cannot register tools, alter MCP side-effect classification or remove confirmation requirements.

- [ ] **Step 1: Write policy repository tests**
  - Test default baseline, candidate creation, parent/version linkage, promotion only with confirmation, active-version lookup and rollback.
  - Reject unknown keys, Python/Shell/code fields, secret-like values, invalid prompt references, memory bounds/filters, output constraints, checker IDs, unknown tool names, more than 3 rounds and empty rationale.
  - Assert failed promotion and rollback leave the active policy unchanged.

- [ ] **Step 2: Run tests to verify failure**
  - Run: .venv\Scripts\python.exe -m pytest tests/edgemind/test_agent_policies.py -q
  - Expected: FAIL because AgentPolicyService is absent.

- [ ] **Step 3: Add schema-v3 policy tables and transactional operations after the v2 feedback migration**
  - Store immutable policy JSON snapshots and parent IDs; keep a single active pointer that changes atomically.
  - Record promotion/rollback history without deleting older policy versions.

- [ ] **Step 4: Implement policy schema validation and service**
  - Accept only prompt_suffix (maximum 4,000 characters) or a registered prompt_template_ref, memory_top_k (0–12), weights for the four named memory layers (0–2), optional time_decay_days (1–3,650), bounded allowed/excluded memory tags, registered tool IDs, max_tool_rounds (1–3), a fixed output_constraints schema, and checker IDs from the built-in checker registry.
  - Reject credential-shaped values and unknown configuration fields; never deserialize executable objects.

- [ ] **Step 5: Run policy persistence tests**
  - Run the command from Step 2.
  - Expected: PASS, including atomic active-policy invariants.

---

### Task 3: Apply Agent Policies Without Expanding Permissions

**Files:**
- Modify: edgemind/harness/agent.py
- Modify: edgemind/harness/memory.py
- Test: tests/edgemind/test_agent_tools.py
- Test: tests/edgemind/test_agent_policies.py

**Interfaces:**
- AgentService accepts optional policy_service: AgentPolicyService | None while preserving existing constructor call compatibility.
- Each run resolves one immutable AgentPolicy snapshot before model invocation and records policy_id/version in its trace.
- The active policy controls prompt suffix, memory top-k/layer weights, built-in tool allowlist and bounded tool rounds.
- Effective tool names are the intersection of current registered/credential-ready tools, policy allowlist, and selected Skill allowlist; MCP confirmation and side-effect classification stay with ToolRegistry.

- [ ] **Step 1: Write failing Agent policy integration tests**
  - Assert selected policy changes memory limit, tool schemas, prompt suffix and max rounds.
  - Assert selected Skill further narrows a broader policy; disabled Brave remains unavailable; unknown policy tools fail closed.
  - Assert MCP side-effect proposal remains pending until explicit confirmation.
  - Assert trace records the exact policy ID/version used even if active policy changes after the run starts.

- [ ] **Step 2: Run focused tests to verify failure**
  - Run: .venv\Scripts\python.exe -m pytest tests/edgemind/test_agent_policies.py tests/edgemind/test_agent_tools.py -q
  - Expected: FAIL because AgentService does not resolve AgentPolicyService.

- [ ] **Step 3: Add memory retrieval weight injection**
  - Extend MemoryService.retrieve with optional validated layer_weights, time_decay_days and tag filters, preserving current defaults and score explanations for all existing callers.

- [ ] **Step 4: Apply the run policy snapshot**
  - Resolve policy once at run start; apply prompt suffix, retrieval values, tool intersection and max rounds; do not apply policy changes midway through a run.
  - Include policy ID/version and safe configuration hash in the run trace.

- [ ] **Step 5: Run Agent policy integration tests**
  - Run the command from Step 2.
  - Expected: PASS; Skills and MCP restrictions continue to take precedence.

---

### Task 4: Versioned Dataset Loader and Isolated Agent Replay

**Files:**
- Create: edgemind/agent_evaluation.py
- Create: tests/data/agent_eval_v1.jsonl
- Test: tests/edgemind/test_agent_evaluation.py
- Reuse: edgemind/harness/agent.py, edgemind/harness/store.py, edgemind/evaluation.py only for shared metric utilities where types match

**Interfaces:**
- Produces load_agent_dataset(path: str | Path) -> AgentDataset.
- AgentDataset has dataset_id, version, description and immutable cases.
- AgentEvaluationRunner(agent_factory, fixture_tool_factory, temp_store_factory) exposes run(dataset_path: str | Path, policy: AgentPolicy, mode: str = "fixture", repetitions: int = 1) -> dict.
- mode is fixture or live; fixture mode requires explicit provider/tool fixtures and never reads live credentials. Live mode is only called by an explicit user action.
- Every report includes dataset hash/version, policy snapshot hash/version, mode, provider/model metadata when available, repetitions, metrics, N/A reasons, case results and error categories.

- [ ] **Step 1: Write failing loader and isolation tests**
  - Dataset cases cover task continuation, memory usefulness, allowed/forbidden tool selection, MCP confirmation boundary, RAG citation expectation and provider/tool failure recovery.
  - Reject invalid JSONL, duplicate example IDs, unsupported schema version, dynamic-code fields, secret-like values and oversized cases.
  - Assert fixture evaluation writes only to temporary HarnessStore and leaves the supplied production store unchanged.

- [ ] **Step 2: Run tests to verify failure**
  - Run: .venv\Scripts\python.exe -m pytest tests/edgemind/test_agent_evaluation.py -q
  - Expected: FAIL because dataset loader and AgentEvaluationRunner are absent.

- [ ] **Step 3: Add a versioned JSONL manifest and scenario cases**
  - Store a manifest record followed by cases; include expected tool names, forbidden tool names, optional required memory/citation IDs, fixture responses, fixture tool outputs and deterministic check definitions.
  - Keep the included scenario set small and label it as a regression suite, not a statistically representative benchmark.

- [ ] **Step 4: Implement strict dataset loading**
  - Validate manifest, version, case IDs, bounded inputs/fixture sizes, allowed checker names and secret/code exclusions before creating any temporary task.

- [ ] **Step 5: Implement isolated fixture replay**
  - Create a fresh temporary HarnessStore per evaluation; execute each case through AgentService with injected provider and tool fixtures; collect run traces and deterministic check results.
  - Ensure fixture mode never launches MCP servers or invokes Brave/Open-Meteo/network clients.

- [ ] **Step 6: Compute metrics with explicit N/A**
  - Compute deterministic fixture-checker metrics (allowlist compliance, tool call success, expected-result checks, memory precision/recall only with gold IDs, citation coverage only with required citation IDs). Report user-annotated task outcomes and memory helpfulness as a separate metric family; fixture runs do not fabricate user feedback. Report latency/token usage only when available.
  - Record repeated-run mean and dispersion; do not treat missing values as zero.

- [ ] **Step 7: Run evaluation tests**
  - Run: .venv\Scripts\python.exe -m pytest tests/edgemind/test_agent_evaluation.py -q
  - Expected: PASS; fixture and live mode labels cannot be confused.

---

### Task 5: Candidate Comparison, Quality Gates, Promotion and Rollback

**Files:**
- Create: edgemind/agent_experiments.py
- Modify: edgemind/harness/store.py
- Test: tests/edgemind/test_agent_experiments.py

**Interfaces:**
- Produces AgentQualityGates(required_min: dict[str, float], max_regression: dict[str, float], max_latency_ms: float | None = None).
- Produces AgentExperimentService(policy_service: AgentPolicyService, evaluation_runner: AgentEvaluationRunner, store: HarnessStore, gates: AgentQualityGates).
- evaluate_candidate(policy_id: str, dataset_path: str | Path, mode: str = "fixture", repetitions: int = 1) -> dict compares candidate and baseline on the same dataset and stores a report.
- promote(experiment_id: str, confirmed: bool) -> AgentPolicy; rollback(policy_id: str, confirmed: bool) -> AgentPolicy.
- A passed gate is necessary but not sufficient: confirmed must be true for promotion/rollback.

- [ ] **Step 1: Write failing comparison/gate tests**
  - Test same-dataset baseline/candidate comparison, required minimums, maximum regressions, latency budget and mandatory zero unsafe/forbidden tool calls.
  - Missing baseline metrics or N/A required metrics block promotion with explicit reasons.
  - Evaluator exception records a failed experiment without changing active policy.
  - Reject unconfirmed promotion/rollback; successful rollback points to an existing version.

- [ ] **Step 2: Run tests to verify failure**
  - Run: .venv\Scripts\python.exe -m pytest tests/edgemind/test_agent_experiments.py -q
  - Expected: FAIL because AgentExperimentService is absent.

- [ ] **Step 3: Implement persisted experiment reports**
  - Add the v3→v4 migration for experiment reports. Save baseline/candidate policy IDs, dataset hash, run mode, deterministic checker metrics, user-feedback metrics when explicitly labeled, per-case differences, gate results and timestamps; cap serialized output and exclude secrets.

- [ ] **Step 4: Implement deterministic gate evaluation**
  - Any unsafe/forbidden tool attempt blocks promotion; compare only metrics with valid numeric values; missing configured gates block instead of passing.

- [ ] **Step 5: Implement explicit promotion and rollback**
  - Promote only a passed report with confirmed=True; switch active policy in the same transaction that records the promotion.
  - Roll back only to an existing policy and require confirmation; append the transition to the policy history.

- [ ] **Step 6: Run experiment tests**
  - Run the command from Step 2.
  - Expected: PASS; blocked, failed and unconfirmed operations never alter active policy.

---

### Task 6: Feedback and Agent Quality UI

**Files:**
- Modify: edgemind/web.py
- Test: tests/edgemind/test_web.py

**Interfaces:**
- Add run outcome controls to the Agent page using the current task/run IDs.
- Add “Agent 质量” page with dataset path/version, fixture/live mode, repetition count, active policy display, candidate policy editor, evaluation report and per-case differences.
- UI callbacks consume QualityService, AgentPolicyService, AgentEvaluationRunner and AgentExperimentService from the existing cached Harness service.
- Live mode requires an explicit button click and a visible confirmation; fixture mode is the default.
- Promotion and rollback buttons require explicit UI confirmation and respect persisted gate status.

- [ ] **Step 1: Write failing callback and component tests**
  - Test recording success/partial/failure and per-memory helpfulness, rendering N/A metrics, displaying blocked gate reasons and hiding promotion for blocked reports.
  - Test UI defaults to fixture mode; selecting live mode exposes an explicit cost/network confirmation.
  - Test no credential values appear in rendered status or report JSON.

- [ ] **Step 2: Run tests to verify failure**
  - Run: .venv\Scripts\python.exe -m pytest tests/edgemind/test_web.py -q
  - Expected: FAIL because the Agent Quality page and callbacks are absent.

- [ ] **Step 3: Add quality page and feedback controls**
  - Keep detailed traces collapsed; show summary outcome, policy version, metrics and direct navigation to failed cases.

- [ ] **Step 4: Wire candidate lifecycle**
  - Connect candidate creation, evaluation, approval, promotion and rollback to service methods; require the matching experiment ID and explicit user confirmation.

- [ ] **Step 5: Run UI tests**
  - Run the command from Step 2.
  - Expected: PASS; existing Agent, memory, tools and diagnostics pages still build.

---

### Task 7: Documentation and Integration Acceptance

**Files:**
- Modify: README.md
- Modify if required: tests and files from Tasks 1–6

**Interfaces:**
- README describes an Agent quality/experience loop, evaluation dataset format, metric definitions, fixture vs live evaluation, policy gates, approval, rollback and project distinction.
- No resume metric or performance claim is added unless supported by a saved report.

- [ ] **Step 1: Document a reproducible local demo**
  - Give commands to run fixture evaluation, annotate a run, create/evaluate a candidate policy, approve promotion and roll back.
  - Document how to add a versioned case and how N/A is computed.

- [ ] **Step 2: Run focused and complete tests**
  - Run: .venv\Scripts\python.exe -m pytest tests/edgemind -q
  - Expected: all tests/edgemind pass; any unrelated legacy root failures must match the baseline recorded before this plan.

- [ ] **Step 3: Start the app and verify the demo path**
  - Start myapp.py; confirm existing Agent/task/media/RAG pages work and the Agent Quality page loads without external keys.
  - Run fixture evaluation and verify its report is marked fixture/mock and contains no credentials.

- [ ] **Step 4: Review safety and scope**
  - Inspect migration, dataset loader, policy schema, feedback redaction, fixture isolation, live evaluation confirmation and MCP approval path.
  - Confirm README still distinguishes EdgeMind from converter-engineering-copilot.

- [ ] **Step 5: Record the acceptance results**
  - Record exact test command/results and note that EdgeMind is not a Git repository, so no commit or Git diff is available.

## Self-Review

- **Spec coverage:** Run ledger and feedback are Task 1; memory utility feedback Tasks 1/6; policy schema and application Tasks 2–3; datasets, isolated replay and metrics Task 4; comparison/gates/promotion/rollback Task 5; UI Task 6; docs/compatibility and acceptance Task 7.
- **Step scan:** Every task begins with a failing test, defines the interface consumed by later tasks, then implements and verifies one reviewable deliverable.
- **Migration consistency:** v1→v2 adds feedback; v2→v3 adds policies and active/history state; v3→v4 adds Agent experiment reports
- **Type consistency:** AgentPolicyService is shared by AgentService and AgentExperimentService; AgentEvaluationRunner consumes an immutable AgentPolicy snapshot and returns deterministic checker reports, with user feedback metrics kept separate; AgentExperimentService persists reports in schema v4; UI uses the same service signatures.
- **Review Focus:** All five high-risk conditions have regression cases assigned above.
- **Proportion:** Seven tasks follow the single feedback→candidate→evaluation→promotion lifecycle; RAG internals and converter-specific features remain out of scope.
