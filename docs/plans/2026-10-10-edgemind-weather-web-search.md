
# EdgeMind 天气、网页搜索与受限 Skills 实施计划


**Goal:** 在 EdgeMind 中交付可真实调用的天气、网页搜索与受限网页读取，配套有边界的 DeepSeek 工具调用循环、两个内置 Skills、配置诊断和来源展示。

**Architecture:** 新增独立的 Open-Meteo、Brave Search、受限网页读取适配器，并由 BuiltinToolRegistry 暴露经参数校验的工具 schema。AgentService 将内置只读工具接入最多 3 轮的 tool-call 循环；MCP 仍由现有 registry 管理，有副作用调用需确认。静态 Skill 清单只编排允许工具，不执行动态代码；Gradio 负责选择、状态与来源展示。

**Tech Stack:** Python 3.11+、现有 OpenAI-compatible DeepSeek SDK、Gradio 6、SQLite/现有 Harness 事件；HTTP 客户端采用 httpx（直接声明轻量运行依赖），pytest 与 monkeypatch 做离线验证。

**Spec:** docs/specs/2026-10-10-edgemind-weather-web-search.md

## Global Constraints

- 运行时保持 Python >=3.11。
- 只从服务端环境读取 BRAVE_SEARCH_API_KEY；任何事件、日志、响应和前端状态不得含 key 或请求头。
- DeepSeek 单轮最多执行 3 个工具调用轮次；不得无限重试或无限循环。
- 只允许显式注册的内置工具和启用的 MCP 工具；拒绝未知名称、非 JSON 对象参数和超出 schema 的参数。
- MCP 写入、删除、外发和代码执行类工具必须经过现有人工确认。
- 页面读取仅限本轮搜索结果或用户本轮给出的 HTTP(S) URL；阻止本机/私有/保留地址、非 HTTP(S)、未复核的重定向目标、超时和超大正文。
- Skill 声明为静态 JSON 数据；不得加载 Skill 中的 Python、Shell 或动态代码。
- 沿用现有 Harness SQLite 表与事件记录；本计划不引入破坏性迁移。
- Open-Meteo 仅用于个人非商业场景，保留 Open-Meteo/GeoNames 数据归属。
- 所有线上 API 测试默认使用 mock；真实 Brave/DeepSeek 请求只在用户配置凭证后做明确的人工冒烟验证。

## Review Focus

1. Brave key 缺失、无效、限流或网络超时：确认未配置工具不发送给模型，且错误信息不泄露凭证。由 Task 1/Task 2 的 Brave client 测试覆盖。
2. 天气地点无匹配或有多个同名地点、外部 API 返回异常：确认不猜地点且给出可澄清结果。由 Task 1 的天气测试覆盖。
3. 网页 URL 指向 localhost、私网、保留地址，或经重定向到受限目标：确认请求在连接前被拒绝。由 Task 1 的 fetch 测试覆盖。
4. 模型返回畸形 JSON、未知工具、多个工具调用或超过 3 轮：确认拒绝/停止并记录可解释轨迹。由 Task 3 的 Agent 测试覆盖。
5. MCP 有副作用工具与内置只读工具出现在同一轮：确认内置工具遵循边界，MCP 在批准前不执行。由 Task 3 的 Agent 测试覆盖。

---

### Task 1: Weather、Search 和 Safe Fetch 适配器

**Files:**
- Create: edgemind/harness/integrations/__init__.py
- Create: edgemind/harness/integrations/weather.py
- Create: edgemind/harness/integrations/search.py
- Create: edgemind/harness/integrations/fetch.py
- Modify: pyproject.toml
- Modify: uv.lock
- Test: tests/edgemind/test_weather.py
- Test: tests/edgemind/test_search.py
- Test: tests/edgemind/test_fetch.py

**Interfaces:**
- Weather produces OpenMeteoWeatherClient.get_weather(location: str, mode: str = "current", forecast_days: int = 1) -> dict.
- Search produces BraveSearchClient(api_key: str | None = None, timeout_s: float = 8.0).search(query: str, count: int = 5, country: str = "CN", search_lang: str = "zh-hans", freshness: str | None = None) -> dict.
- Fetch produces SafePageFetcher(timeout_s: float = 10.0, max_bytes: int = 2 * 1024 * 1024).fetch(url: str, allowed_urls: set[str]) -> dict.
- Clients return normalized JSON-safe dictionaries or typed, non-secret error categories; tests inject/mock HTTP transport and never call the public Internet.

- [x] Step 1: Add failing tests for weather geocoding, multiple-match disambiguation, current/forecast normalization, API failure, and attribution.
- [x] Step 2: Run tests/edgemind/test_weather.py and confirm the new client contract fails.
- [x] Step 3: Implement OpenMeteoWeatherClient with bounded requests, location choice metadata, local-time labeling, and a short TTL cache keyed by location/mode/days.
- [x] Step 4: Run tests/edgemind/test_weather.py and confirm all weather cases pass.
- [x] Step 5: Add failing tests for Brave key missing, query/count validation, normalized links/snippets, 401/429/timeout, and secret redaction.
- [x] Step 6: Implement BraveSearchClient using the official API and server-side BRAVE_SEARCH_API_KEY.
- [x] Step 7: Add failing SafePageFetcher tests for allowed URL success, private/local addresses, redirect revalidation, MIME rejection, byte limit, and timeout.
- [x] Step 8: Implement SafePageFetcher with redirects disabled by default and manually validated per hop; resolve and reject non-public targets before each connection; accept only HTML/plain text and strip markup.
- [x] Step 9: Add httpx as a direct dependency, update uv.lock, then run all three adapter test modules offline.

### Task 2: Built-in Tool Registry and Availability Diagnostics

**Files:**
- Create: edgemind/harness/builtin_tools.py
- Modify: edgemind/harness/__init__.py
- Test: tests/edgemind/test_builtin_tools.py

**Interfaces:**
- BuiltinToolRegistry(weather: OpenMeteoWeatherClient | None = None, search: BraveSearchClient | None = None, fetcher: SafePageFetcher | None = None) exposes model_tools() -> list[dict], execute(function_name: str, arguments: dict, allowed_urls: set[str] | None = None) -> dict, and status() -> list[dict].
- Registered model names are edgemind__get_weather, edgemind__web_search, and edgemind__fetch_search_result.
- model_tools() omits Brave and fetch tools when BRAVE_SEARCH_API_KEY is missing; weather remains available. status() returns only readiness, service label, and safe diagnostic text.

- [x] Step 1: Add failing tests for exact schemas, missing-key tool omission, invalid arguments, unknown tool rejection, and status redaction.
- [x] Step 2: Run tests/edgemind/test_builtin_tools.py and confirm the registry contract fails.
- [x] Step 3: Implement BuiltinToolRegistry as an explicit dispatcher; validate types, allowed enums, lengths, count bounds, and 1–7 forecast days before calling adapters.
- [x] Step 4: Export the registry from edgemind.harness and run the registry tests.
- [x] Step 5: Run adapter and registry modules together to verify adapters remain independently usable.

### Task 3: Bounded DeepSeek Tool Loop and MCP Approval Boundary

**Files:**
- Modify: edgemind/harness/agent.py
- Modify: edgemind/harness/tools.py
- Test: tests/edgemind/test_agent_tools.py
- Test: tests/edgemind/test_harness.py

**Interfaces:**
- AgentService constructor accepts builtin_tools: BuiltinToolRegistry | None.
- run_turn gains optional skill_id: str | None = None and on_tool_status: Callable[[str], None] | None = None, preserving current positional arguments.
- Tool loop combines BuiltinToolRegistry.model_tools() with existing MCP model tools; uses stable tool_call_id values when returning tool messages.
- Each tool-call round executes registered built-in read-only tools, persists non-secret tool events and elapsed time, and calls DeepSeek again. Stop after exactly 3 tool-call rounds and request a final answer from collected results without offering more tools.
- Existing MCP proposal IDs continue through HarnessStore. Side-effecting MCP calls create proposals and are not executed; confirmation resumes with the matching tool result. Invalid calls return bounded structured errors without exposing stack traces or credentials.
- Trace adds tool_calls (name, status, elapsed_ms, result_count/error_type, source metadata) and tool_rounds; never stores auth headers.

- [x] Step 1: Add FakeClient tests for one built-in call + final synthesis, multiple calls in one response, a second tool round, and max-round stop.
- [x] Step 2: Add failure tests for malformed JSON, unknown tool, tool exception, and MCP side-effect proposal that remains unexecuted pending confirmation.
- [x] Step 3: Run tests/edgemind/test_agent_tools.py and confirm the loop behavior fails before implementation.
- [x] Step 4: Refactor AgentService.run_turn into a bounded message/tool loop while preserving memory, RAG, media, checkpoints, and pending MCP proposal output.
- [x] Step 5: Add tool-call result events and per-call timings; redact errors and cap persisted result text according to the spec.
- [x] Step 6: Ensure execute_tool_proposal resumes the exact matching tool call after confirmation and returns a final answer.
- [x] Step 7: Run Agent tool tests and existing tests/edgemind/test_harness.py.

### Task 4: Static Skills Registry and Two Workflows

**Files:**
- Create: edgemind/harness/skills.py
- Create: edgemind/harness/skills/weather_briefing.json
- Create: edgemind/harness/skills/web_research.json
- Modify: edgemind/harness/agent.py
- Test: tests/edgemind/test_skills.py
- Test: tests/edgemind/test_agent_tools.py

**Interfaces:**
- SkillRegistry.list_skills() -> list[SkillDefinition], get_skill(skill_id: str) -> SkillDefinition | None.
- SkillDefinition contains id, version, title, description, allowed_tools, max_tool_rounds, instructions, and input_schema.
- weather_briefing allows only edgemind__get_weather; web_research allows edgemind__web_search and edgemind__fetch_search_result and caps page fetches at 3.
- Explicitly selected skills restrict the tool schemas and append their instructions; unknown skills fail validation before any model request. No skill selected preserves general built-in tool use.
- Trace records selected skill id/version; skills are not granted extra permissions.

- [x] Step 1: Add failing tests for loading both declared skills, invalid JSON/schema, unknown skill ID, restricted tool exposure, and 3-page research cap.
- [x] Step 2: Run tests/edgemind/test_skills.py and confirm the registry contract fails.
- [x] Step 3: Implement SkillRegistry using JSON data only; validate schema and version during load and return immutable SkillDefinition values.
- [x] Step 4: Integrate skill selection into AgentService.run_turn; reject tool calls outside the selected skill allowlist.
- [x] Step 5: Run Skills tests and Agent tool tests, including that no dynamic code is imported.

### Task 5: Gradio Tool UX, Configuration Health, and User Documentation

**Files:**
- Modify: edgemind/web.py
- Modify: README.md
- Create: .env.example
- Test: tests/edgemind/test_web.py

**Interfaces:**
- run_agent_turn and agent_chat_turn accept selected skill ID and pass it to AgentService.run_turn.
- A tool-status helper returns statuses for Open-Meteo, Brave Search, and configured MCP without secrets.
- Manual health check performs one bounded request only when the user clicks; it never exposes credentials.
- The existing MCP page becomes an Agent tools page with built-in/MCP readiness, Brave configuration guidance, health-check action, and approval controls; the composer exposes a skill dropdown and shows recent tool state in the run trace.
- on_tool_status updates a Gradio progress description while each tool runs; final messages show clickable source links.

- [x] Step 1: Add failing UI helper tests for Brave key ready/missing states, secret redaction, skill choices, and health-check result formatting.
- [x] Step 2: Run tests/edgemind/test_web.py and confirm these UI contracts fail.
- [x] Step 3: Implement tool readiness and explicit health-check helpers in edgemind/web.py.
- [x] Step 4: Add the skill selector and progress/status plumbing; revise MCP page labels and display tool sources/traces in the current GPT-style layout.
- [x] Step 5: Update README.md and .env.example with Brave setup, Open-Meteo attribution, workflows, tool permissions, and troubleshooting.
- [x] Step 6: Run tests/edgemind/test_web.py and all tests/edgemind modules; confirm the legacy RAG import remains lazy and existing task/media flows still load.

### Task 6: Integration Review and Manual Acceptance

**Files:**
- Modify if needed: files identified by Tasks 1–5
- Test: existing and new tests in tests/edgemind

- [x] Step 1: Run the complete offline test suite and record the exact command and result in the implementation summary.
- [x] Step 2: Start the application with no Brave key and verify it starts, displays “not configured,” and text/RAG/task pages remain available.
- [x] Step 3: When the user has configured BRAVE_SEARCH_API_KEY, use the UI health check once and verify search returns clickable sources without exposing the key.
- [x] Step 4: Verify a weather query through the UI returns location and local time; if live network access is unavailable, record the limitation and rely on mocked contract tests.
- [x] Step 5: Review the diff for credential leakage, unintended DB migrations, MCP confirmation regressions, and divergence from converter-engineering-copilot.

## Self-Review

- **Spec coverage:** Tasks 1–2 implement weather, Brave Search, fetch, and availability; Task 3 implements the bounded call loop, safety, traces, MCP boundary, and latency; Task 4 implements Skills; Task 5 implements UI, status, health checks, and documentation; Task 6 covers acceptance and compatibility.
- **Step scan:** Each task separates failing tests, minimal implementation, and verification; public names and bounds are fixed above. No step leaves retry count, tool limit, result fields, or settings source to the implementer.
- **Type consistency:** Registry identifiers are edgemind__get_weather, edgemind__web_search, and edgemind__fetch_search_result across schema, Skills, Agent and tests. SkillDefinition fields match the registry contract. AgentService signature preserves old positional calls.
- **Review Focus:** All five highest-risk input classes map to explicit adapter/registry/Agent tests listed above.
- **Proportion:** Six sequential tasks map directly to the spec; no database migration or unrelated refactor is included.
