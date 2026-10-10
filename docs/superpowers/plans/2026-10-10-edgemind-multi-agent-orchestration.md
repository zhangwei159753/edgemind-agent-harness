# EdgeMind Bounded Multi-Agent Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, observable multi-agent workflow for complex tasks while preserving the existing single-agent path and tool safety guarantees.

**Architecture:** Keep `AgentService.run_turn()` as the public compatibility entry point and delegate selected complex turns to a separate orchestration module. A coordinator executes Planner, Researcher, Executor, and Reviewer roles sequentially with versioned prompts, validated handoff data, shared provider configuration, fixed budgets, and event traces.

**Tech Stack:** Python 3.11+, existing DeepSeek/OpenAI-compatible client, SQLite `HarnessStore`, Gradio, pytest; no new framework dependency.

**Spec:** `docs/specs/2026-10-10-edgemind-multi-agent-memory-graph-tot.md`

## Global Constraints

- 首版采用单协调器加职责明确的逻辑 Agent，共享现有 Provider/模型配置；不要求多模型或新增模型 API。
- 首版按顺序执行 Agent handoff，使用有版本的角色提示、输入/输出契约和独立能力范围。并行执行留给后续只读、互相独立的子任务。
- 简单任务保留单 Agent；多步骤、需要多个信息源或明确计划的任务才进入多 Agent 编排。
- 提示、网页内容、记忆和工具输出均视为不可信数据，不能升级成系统指令。
- 工具权限受策略与工作流 allowlist 约束；MCP 副作用调用仍需人工确认。

## Review Focus

- 简单请求被误路由到多 Agent：Task 1 必须测试单步请求仍走既有路径。
- Planner 返回格式损坏或越界字段：Task 1 必须测试拒绝错误 handoff 且不执行工具。
- 复杂任务需要跨角色传递证据：Task 2 必须测试 handoff 只传 evidence/memory ID 与限定摘要。
- MCP 副作用工具经过 Agent fallback：Task 2 必须测试任何失败路径都不绕开待确认状态。
- Reviewer 发现证据缺口或达到重试上限：Task 3 必须测试至多一次修订并返回明确未完成项。

---

### Task 1: Orchestration contracts and routing

**Files:**
- Create: `edgemind/harness/orchestration/models.py`
- Create: `edgemind/harness/orchestration/routing.py`
- Create: `tests/edgemind/test_agent_orchestration_models.py`
- Create: `tests/edgemind/test_agent_orchestration_routing.py`

**Interfaces:**
- Produce `Handoff(run_id: str, task_id: str, role: str, goal: str, evidence_ids: tuple[str, ...], allowed_tools: tuple[str, ...], remaining_calls: int, output_schema: str)` and `RoleResult(role: str, status: str, payload: dict, error_type: str | None)` immutable dataclasses.
- Produce `should_orchestrate(task_objective: str, user_input: str, mode: str = "auto") -> bool`; accepted modes are `auto`, `single`, `multi`. `single` always returns false and `multi` always returns true. `auto` returns true only when input contains a sequence marker (`先…再…`, `然后`, `接着`, `依次`) or at least two requested action verbs (`搜索`, `查询`, `检索`, `比较`, `核验`, `总结`, `分析`, `整理`, `生成`) connected by `并/并且/以及`; otherwise it returns false. These routing modes never override security policy.
- Handoff validation rejects unknown fields, missing IDs/goal, unsupported role/schema, overlong strings/collections, and non-positive budgets.

- [ ] **Step 1: Write failing tests** for valid/invalid `Handoff`, single-step auto routing, multi-step auto routing, and explicit overrides.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_agent_orchestration_models.py tests/edgemind/test_agent_orchestration_routing.py -q`; confirm expected import/behavior failures.
- [ ] **Step 3: Implement** immutable contracts and deterministic router in the two new modules; add no provider calls or persistence here.
- [ ] **Step 4: Run the same tests**; expect all route and schema boundary cases to pass.
- [ ] **Step 5: Commit** as `feat: add orchestration contracts and routing`.

### Task 2: Sequential role runner and safe handoffs

**Files:**
- Create: `edgemind/harness/orchestration/runner.py`
- Create: `edgemind/harness/orchestration/prompts.py`
- Modify: `edgemind/harness/store.py` (event retrieval only if current API cannot return run events)
- Test: `tests/edgemind/test_agent_orchestration.py`

**Interfaces:**
- `OrchestrationRequest(task_id: str, run_id: str, objective: str, user_input: str, mode: str, memory_hits: tuple, evidence: tuple, allowed_tools: tuple[str, ...])`.
- `ExecutorResult(answer: str, status: str, pending_tool_calls: list[dict], trace: dict)`.
- `AgentOrchestrator(client, store, role_call_budget=8, max_steps=5, max_revisions=1)` with `run(request: OrchestrationRequest, executor: Callable[[Handoff, list[dict]], ExecutorResult], stage_ms: dict[str, float]) -> OrchestrationResult`.
- `_invoke_role(role: str, handoff: Handoff) -> RoleResult` uses the shared chat completion client for Planner/Reviewer, requires a JSON object in the response, and validates it against the role schema; Researcher is a deterministic retrieval adapter and does not make a model call.
- The executor callback receives the current validated handoff and bounded chat messages; the AgentService adapter invokes the existing `_run_tool_loop()` and maps its return tuple to `ExecutorResult`. It is the only route to built-in/MCP execution and must preserve proposal/confirmation semantics.
- `OrchestrationResult(answer: str, status: str, trace: dict, selected_plan: dict | None, pending_tool_calls: list[dict])`.

- [ ] **Step 1: Write failing fixture tests** for expected role order, bounded payload handoff, evidence-ID preservation, role-call budget, provider errors, malformed output, and no MCP execution on fallback.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_agent_orchestration.py -q`; confirm behavior failures without network calls.
- [ ] **Step 3: Implement** sequential role runner; version prompts; persist only structured handoff/plan/review summaries using `HarnessStore.append_event`; store no hidden CoT/scratchpad. Validate each role payload and cap serialized handoff messages at 60 KiB.
- [ ] **Step 4: Run the tests** and assert executor invocation remains allowlist-filtered and side effects remain pending confirmation.
- [ ] **Step 5: Commit** as `feat: add bounded sequential agent orchestration`.

### Task 3: AgentService and UI integration

**Files:**
- Modify: `edgemind/harness/agent.py`
- Modify: `edgemind/web.py`
- Test: `tests/edgemind/test_agent_tools.py`
- Test: `tests/edgemind/test_web.py`
- Create: `tests/edgemind/test_agent_orchestration_integration.py`

**Interfaces:**
- Keep the existing `AgentService.run_turn(...) -> AgentTurn` signature compatible; add optional `orchestration_mode="auto"` as a trailing keyword.
- Add `orchestration` trace fields: `mode`, `route_reason`, `handoffs`, `selected_plan_summary`, `review_status`, and `call_count`.
- The UI shows mode and role-stage status in the existing trace panel; no full hidden reasoning text is exposed.

- [ ] **Step 1: Write failing integration tests** for legacy callers, simple auto route, explicit multi route, trace serialization, and awaiting-tool-approval result.
- [ ] **Step 2: Run** `python -m pytest tests/edgemind/test_agent_orchestration_integration.py -q`; confirm failures.
- [ ] **Step 3: Inject** `AgentOrchestrator` into `AgentService`, delegate only when routed, preserve current single-agent behavior and `AgentTurn` shape, then expose concise trace in Gradio.
- [ ] **Step 4: Run** `python -m pytest tests/edgemind/test_agent_orchestration_integration.py tests/edgemind/test_agent_tools.py tests/edgemind/test_web.py -q`; expect pass with existing callbacks intact.
- [ ] **Step 5: Commit** as `feat: integrate orchestration with agent workspace`.

### Task 4: Regression, documentation, and full verification

**Files:**
- Modify: `README.md`
- Modify: `tests/data/agent_eval_v1.jsonl` only if additive cases can preserve dataset version compatibility; otherwise create `tests/data/agent_orchestration_v1.jsonl`.
- Test: `tests/edgemind/test_agent_orchestration_integration.py`

- [ ] **Step 1: Add fixture cases** for simple routing, multi-step handoff, provider failure, and reviewer evidence gap.
- [ ] **Step 2: Run targeted tests** and ensure fixture mode makes no external DeepSeek/MCP/network calls.
- [ ] **Step 3: Document** opt-in/auto routing, role boundaries, trace fields, budgets, and safety behavior in README.
- [ ] **Step 4: Run** `python -m pytest tests/edgemind -q`; expect all tests pass and no live API is required.
- [ ] **Step 5: Commit** as `docs: document multi-agent orchestration`.
