# EdgeMind Bounded ToT Planning and Comparative Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a constrained Tree-of-Thought-style plan selector and prove its value against the single-agent baseline using isolated, fixed evaluation fixtures.

**Architecture:** Implement ToT as at most three structured candidate plans with deterministic validation/scoring, not exposed model scratchpad. Feed only the selected plan to the orchestration runner, extend the existing Agent JSONL evaluator for mode/plan assertions and resource metrics, and gate rollout on quality, safety, latency, and call-cost reports.

**Tech Stack:** Python 3.11+, existing DeepSeek-compatible client, existing `AgentEvaluationRunner` and fixture adapters, SQLite, Gradio, pytest; no new dependency.

**Spec:** `docs/specs/2026-10-10-edgemind-multi-agent-memory-graph-tot.md`

## Global Constraints

- ToT 实现为有预算的候选计划搜索，而非展开或展示原始模型思维过程。
- Planner 最多提出 3 个候选计划；首版最多 5 个子步骤、一个执行阶段和一次 Reviewer 修订。
- 安全条件是硬门槛，不可由其它分数抵消；平分时选择步骤更少的方案。
- 默认 Fixture 评测不访问真实 DeepSeek、MCP 或外部网络；真实评测单独标注并由用户显式启动。
- 缺少 gold/usage 时显示 N/A；工具 allowlist 越权数必须为 0。

## Review Focus

- Malformed/extra-field plan output: Task 1 tests strict schema rejection and a safe clarification fallback.
- Unsafe/disallowed or side-effecting proposed tools: Task 1 tests hard rejection before execution and preserves approval flow.
- Candidate tie or all candidates invalid: Task 1 tests deterministic simpler-plan tie break and no unsafe fallback.
- Fixture schema compatibility: Task 2 tests old dataset v1 still loads and new planning fields remain bounded.
- Missing labels/token usage and performance regression: Task 2/3 test N/A semantics and enforce quality/safety/cost report gates.

---

### Task 1: Candidate plan schema, hard constraints, and scorer

**Files:**
- Create: `edgemind/harness/orchestration/planning.py`
- Modify: `edgemind/harness/orchestration/__init__.py` (public planner exports)
- Create: `tests/edgemind/test_agent_tot_planning.py`
- Modify: `edgemind/harness/orchestration/models.py` from the orchestration plan

**Interfaces:**
- Add `PlanStep(step_id: str, goal: str, depends_on: tuple[str, ...], expected_evidence: tuple[str, ...], expected_tools: tuple[str, ...], risk: str, estimated_calls: int)`.
- Add `PlanCandidate(candidate_id: str, steps: tuple[PlanStep, ...], goal_coverage: float, evidence_availability: float, tool_availability: float, risk_score: float, safety_ok: bool, estimated_cost: float)`; normalized numeric scores, risk and cost are finite in [0,1].
- `validate_candidates(raw: object, allowed_tools: set[str], max_steps: int = 5, max_candidates: int = 3) -> tuple[PlanCandidate, ...]` rejects unknown fields, duplicate IDs, cycles, undeclared dependencies, forbidden tools, oversized text, out-of-range values, and unsafe candidates.
- `select_candidate(candidates: tuple[PlanCandidate, ...]) -> tuple[PlanCandidate | None, dict]` ranks only safety-valid plans with fixed weights; ties prefer fewer steps then candidate ID. No candidates returns `(None, reason_summary)`.
- Score formula: `0.35*goal_coverage + 0.25*evidence_availability + 0.20*tool_availability + 0.10*(1-risk_score) + 0.10*(1-estimated_cost)`. Reject `safety_ok=False` before scoring; all inputs are normalized finite values in [0,1].

- [ ] **Step 1: Write failing tests** for schema bounds, graph cycles, forbidden tools, hard safety, no-valid-plan, max-3 candidates and deterministic tie-break.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_agent_tot_planning.py -q`; confirm failures.
- [ ] **Step 3: Implement** pure validation and deterministic scoring with no model/provider calls in the scorer.
- [ ] **Step 4: Run tests** and assert output contains only plan summaries/scores, not raw scratchpad.
- [ ] **Step 5: Commit** as `feat: validate and score bounded plan candidates`.

### Task 2: Planner integration and regression dataset format

**Files:**
- Modify: `edgemind/harness/orchestration/runner.py`
- Modify: `edgemind/agent_evaluation.py`
- Modify: `edgemind/agent_eval_fixtures.py`
- Create: `tests/edgemind/test_agent_tot_integration.py`
- Modify: `tests/edgemind/test_agent_eval_fixtures.py`
- Create: `tests/data/agent_multi_agent_v1.jsonl`

**Interfaces:**
- `AgentOrchestrator` stores `selected_plan` and rejected candidate summaries in the trace, executes only the selected plan, and aborts planning when candidate/call/step budgets are exhausted.
- Add optional case fields `orchestration_mode`, `expected_plan_id`, `expected_roles`, and `max_model_calls` (integer 1..8); preserve existing v1 records by defaulting to single mode and existing limits. Reuse the existing secret/dynamic-code scanner for every new field.
- Fixture `responses` remain deterministic and fixture mode must not execute MCP or external I/O.

- [ ] **Step 1: Write failing tests** for selecting and executing one plan, all-invalid fallback, provider error, max-call abort, expected role sequence, legacy v1 loader compatibility, and MCP non-execution.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_agent_tot_integration.py tests/edgemind/test_agent_eval_fixtures.py -q`; confirm failures.
- [ ] **Step 3: Integrate** validated candidate selection with the prior orchestration runner and add backward-compatible loader fields.
- [ ] **Step 4: Run tests**; verify one candidate is executed, no candidate raw output leaks to final trace, and all fixture APIs stay isolated.
- [ ] **Step 5: Commit** as `feat: integrate bounded planning with agent evaluation`.

### Task 3: Comparative metrics, gate, and quality UI

**Files:**
- Modify: `edgemind/agent_evaluation.py`
- Modify: `edgemind/agent_experiments.py`
- Modify: `edgemind/web.py`
- Create: `tests/edgemind/test_agent_multi_agent_evaluation.py`
- Modify: `tests/edgemind/test_agent_experiments.py`
- Modify: `tests/edgemind/test_web.py`

**Interfaces:**
- Evaluation report adds `mode`, `case_pass_rate`, `citation_coverage`, `allowlist_compliance`, `forbidden_tool_attempts`, `memory_precision_at_k`, `latency_ms_mean`, `model_call_count_mean`, and `token_usage_mean`; unavailable gold/usage produces `None` and UI renders N/A.
- Comparison requires identical dataset hash and reports per-case differences for single vs multi mode.
- Promotion gate rejects any forbidden tool attempt, `case_pass_rate < 0.80`, regression against single-agent baseline greater than 0.05, or measured p95 latency above a user-entered `max_latency_ms`; max planner/model calls remain 8 per run. Add `latency_ms_p95` to the report. Missing gold/usage is N/A and never treated as zero. Token usage is reported when returned by the provider; latency and call-count budgets still apply when token usage is unavailable.

- [ ] **Step 1: Write failing tests** for metric N/A, equal-dataset comparison, per-case output, forbidden-tool hard gate, quality/cost regression gate and no usage provider.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_agent_multi_agent_evaluation.py tests/edgemind/test_agent_experiments.py -q`; confirm failures.
- [ ] **Step 3: Implement** additive report/gate fields and render comparison in the existing Agent Quality view; do not change existing RAG evaluation semantics.
- [ ] **Step 4: Run tests** and inspect the fixture report for clear baseline/candidate mode labels and N/A values.
- [ ] **Step 5: Commit** as `feat: gate multi-agent rollout on regression evidence`.

### Task 4: End-to-end fixture and documentation

**Files:**
- Modify: `tests/data/agent_multi_agent_v1.jsonl`
- Modify: `README.md`
- Test: `tests/edgemind/test_agent_tot_integration.py`
- Test: all `tests/edgemind`

- [ ] **Step 1: Ensure fixture cases** cover simple single-agent path, complex collaboration, evidence gap/revision, unsafe plan, provider failure, and budget exhaustion.
- [ ] **Step 2: Run targeted tests**; verify all multi-agent reports are fixture-mode and no real API/network is used.
- [ ] **Step 3: Document** candidate generation/scoring, caps, comparison metrics, rollout gate and limitations.
- [ ] **Step 4: Run** `python -m pytest tests/edgemind -q`; expect all tests pass.
- [ ] **Step 5: Commit** as `docs: document bounded planning evaluation`.
