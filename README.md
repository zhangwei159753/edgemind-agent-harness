# EdgeMind — Personal Agent Harness

EdgeMind 是一个本地优先的个人 Agent 长期任务与记忆运行时。它把多模态输入、跨会话任务 checkpoint、可审核的四层记忆、个人知识库 RAG 和受控 MCP 工具调用放进一个可观测的 Harness。

它与 converter-engineering-copilot 的边界是：EdgeMind 解决通用个人 Agent 的任务续接、记忆治理和工具运行时；Converter Copilot 继续解决电力变换器研发/运维中的专业分析与证据工作流。

## 能力

- 在本机创建、暂停、恢复和完成长程任务；任务目标、阶段状态和最近 checkpoint 存在 SQLite。
- 四层记忆：Working（当前任务）、Episodic（经历）、Semantic（事实/偏好）、Procedural（经验/流程）。长期记忆候选默认待审核，批准后才进入正常检索。
- 记忆检索优先使用可解释的 SQLite FTS/BM25 排序；配置本地 embedding 服务后，语义相似度参与融合，服务不可用时退回词法检索。
- 文本与图片输入直接进入 DeepSeek Chat Completions；音频通过可选本地 faster-whisper 转写。
- 可通过官方 MCP Python SDK 使用本地 stdio MCP servers；工具列表可控，有副作用的调用必须在界面确认。
- RAG 回答保留证据、引用、运行轨迹、语料版本、离线评测、反馈和配置晋升/回滚门禁。
- Gradio 本地交互台包含 Agent 工作区、记忆审核、内置天气/搜索与 MCP 工具状态、手动连通检查、DeepSeek API 诊断和 RAG 诊断。

## 安装与启动

Python 3.11+。用现有基础依赖启动 Web 界面：

```powershell
uv sync
Copy-Item .env.example .env
# 编辑 .env，至少设置 DEEPSEEK_API_KEY
python myapp.py
```

浏览器打开 http://127.0.0.1:7860。也可运行 `python -m edgemind --help` 查看 RAG 评测和治理命令。基础安装不拉取旧版 PDF 解析、Notebook 或本地 RAG 模型依赖；需要使用兼容的混合 RAG 时，另执行 `uv sync --extra rag`。

### DeepSeek 图片输入

当前默认模型 deepseek-flash 支持图像与文本的多模态输入。Agent 页可上传 JPEG、PNG、GIF 或 WebP；单图上限 20 MiB。图片只发送给用户配置的 DeepSeek endpoint，不会作为原始字节写入 EdgeMind SQLite。

### 音频转写（可选）

```powershell
uv sync --extra audio
```

默认使用 faster-whisper small、CPU、int8，可在 .env 设置 EDGEMIND_WHISPER_MODEL、EDGEMIND_ASR_DEVICE 和 EDGEMIND_ASR_COMPUTE_TYPE。模型首次加载可能需要下载，建议提前准备好本机模型缓存。单个音频文件上限 40 MiB。未安装 ASR extra 时，界面会返回安装提示，不会伪造转写。

### MCP 工具（可选）

将 `docs/mcp-servers.example.json` 复制为本机 `mcp_servers.json`，按需设置 enabled、command、args 和工具允许列表。将带凭证的环境变量写成 `${ENV_NAME}` 引用，并在启动 EdgeMind 的 shell 环境中设置对应变量。配置示例默认禁用。安装 MCP 客户端依赖：

```powershell
uv sync --extra mcp
```

工具需要由“发现已启用工具”读取。模型提出调用后，Agent 页轨迹会显示 proposal ID、工具和参数；复制 ID 后确认执行或拒绝。工具只有列入配置后才可调用；未列在 read_only_tools 中的工具按有副作用处理。

### 天气、网页搜索与工作流

Agent 工具页显示 Open-Meteo、Brave Search 和 MCP 的可用状态。天气查询使用 Open-Meteo Forecast/Geocoding 服务，无需 API key，适用于个人非商业场景；天气及地理编码数据分别注明 Open-Meteo 与 GeoNames 归属。地点有歧义时会要求澄清。

网页搜索使用 Brave Search API。先在 Brave Search API Dashboard 创建 API key，再将密钥写入本机 .env：

    BRAVE_SEARCH_API_KEY=your_brave_search_api_key

重启 EdgeMind 后，在“Agent 工具”页刷新状态并点击“执行一次检查”验证连接。密钥只从服务端环境读取；UI、运行轨迹与日志不显示密钥。未配置 Brave key 时，天气、任务、记忆和 RAG 仍可使用，网页搜索与网页读取会隐藏/不可用。健康检查会发送一次最多返回 1 条结果的 Brave 查询；天气检查查询北京当前天气。

聊天工作流可选“天气与出行建议”或“主题资料调研”；不选时为普通对话，自动使用已配置工具。工作流是版本化 JSON 声明，不会加载任意 Python/Shell。天气工作流只允许天气工具；调研工作流最多读取 3 个本轮搜索结果。网页读取只限用户本轮给出的公开 URL 或本轮搜索结果，并拒绝本机/私网地址、非网页协议及超大正文。只读内置工具自动执行；MCP 副作用工具仍需用户确认。

故障排查：Brave 显示未配置时检查 .env 文件名、变量拼写并重启进程；认证失败时检查 Brave key 权限/额度；超时或限流时稍后重试。天气无结果或地点候选不唯一时，补充国家、省份或城市全名。

### 有界多 Agent 协作

Agent 工作台提供“自动（默认）”“单 Agent”“多 Agent 协作”三种路由。自动模式只把多阶段任务交给协作流程，普通问答仍走现有单 Agent；显式多 Agent 适合需要先计划、再执行并检查结果的请求，会增加模型调用与延迟。

协作流程由一个协调器按顺序调度 Planner、Researcher、Executor、Reviewer。Planner 最多输出 3 个、每个最多 5 步的候选计划；系统校验 schema、依赖 DAG、证据 ID、工具 allowlist 和调用预算，再按目标覆盖、证据/工具可用性、风险与成本进行确定性评分。安全条件是硬门槛，平分时选步骤更少的计划。Researcher 只整理已有记忆与 RAG 证据 ID，不调用模型；Executor 只获得经选中计划、策略和工作流 allowlist 交集过滤的工具；Reviewer 检查结论与证据，最多触发一次修订。每轮模型调用预算最多 8。Provider 暂时失败可回退到单 Agent；格式错误或不安全计划会停止执行并请求澄清。MCP 副作用仍需人工确认。

运行详情展示路由原因、角色 handoff、选中计划 ID、计划摘要、候选评分摘要、审核状态和模型调用数，不保存或展示隐藏 CoT。基础回归样例位于 `tests/data/agent_orchestration_v1.jsonl`；ToT/评测样例位于 `tests/data/agent_multi_agent_v1.jsonl`。固定 Fixture 不连接 DeepSeek、MCP 或外网：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\edgemind\test_agent_tot_planning.py tests\edgemind\test_agent_tot_integration.py tests\edgemind\test_agent_orchestration.py tests\edgemind\test_agent_orchestration_integration.py -q
```

## 任务与记忆

1. 在 Agent 工作台输入任务名称和目标，创建任务并保留任务 ID。
2. 发送文字、图片或音频；Agent 从该任务 checkpoint 和已批准的相关记忆构造上下文。
3. 暂停任务后可重启应用，再用相同任务 ID 恢复；任务状态、运行事件和 checkpoint 保存在 `.data/edgemind.sqlite3`。
4. 在“记忆审核”页检查待审核项，批准、编辑后批准、拒绝或删除。pending 内容不会进入正常 Agent 上下文。
5. 每轮对话会留下 episodic 候选供用户检查。系统不会静默把用户纠错直接升级成长期事实。
6. 对回答提交纠错后，系统会记录纠错来源、创建待审核 semantic 记忆候选，并用这条纠错重新生成回答。

记忆可导出为 JSONL：

```powershell
python -m edgemind --state .data/edgemind.sqlite3 memory export --output .data/memories.jsonl --approved-only
```

四层定义：

| 层 | 内容 | 生命周期 |
|---|---|---|
| Working | 当前任务的阶段、下一步、阻塞项 | 随任务 checkpoint 更新 |
| Episodic | 有任务来源的交互和行动结果 | 候选审核后参与长期检索 |
| Semantic | 稳定事实、偏好和长期目标 | 用户审核后参与检索 |
| Procedural | 经验证的可复用经验和步骤 | 用户审核后参与检索 |

### 记忆关系图

记忆图使用现有四层记忆作为节点，关系边存入 SQLite；不引入图数据库。支持 `derived_from`、`supports`、`contradicts`、`used_in` 和 `similar_to`。关系必须先创建为待审核边；只有两个端点记忆都已批准、且用户批准关系后，关系才可参与检索。拒绝关系不会参与扩展；删除记忆时关联边由 SQLite 外键级联删除。`similar_to` 会规范端点顺序，避免反向重复。

Agent 工作台默认启用最多 12 个节点、单跳扩展；公共 `MemoryService.retrieve()` 默认保持旧行为，可显式传 `graph_expansion=True` 获取图扩展。应用侧可用 `AgentService(..., memory_graph_expansion=False)` 关闭。相关节点仍受当前任务范围与 memory tag allow/deny 过滤约束。关系分数使用种子记忆分数 × 边置信度 × 关系权重；`contradicts` 只生成冲突轨迹，不作为正向候选。运行轨迹保存关系 ID、类型和分数组件，不保存记忆全文。

在“记忆管理 → 记忆关系图”中填写来源/目标记忆 ID、关系与置信度，创建待审核边；在列表中检查后，用边 ID 批准或拒绝。schema v5 会在应用启动时从旧版本向前追加关系表，保留已有任务、记忆、反馈数据。`tests/data/memory_graph_v1.jsonl` 是隔离 SQLite 夹具，不会触碰真实 `.data/edgemind.sqlite3`。

## Agent 质量与经验闭环

EdgeMind 将“自进化”限定为有证据和人工治理的策略改进流程：记录运行结果与记忆效用反馈，创建不可变策略候选，在固定回归集上和其父策略对比，通过安全/质量门禁后由用户明确晋级。系统不会自行训练模型、改源码、自动修改已批准记忆或无限循环运行。

### 本地演示流程

```powershell
python myapp.py
```

1. 在“Agent 工作台”选择或创建任务，发起一轮对话；展开“结果反馈”，记录 success、partial 或 failure。可选地用 JSON 将本轮实际暴露的记忆 ID 标为 helpful、not_helpful 或 uncertain，例如 `{"memory-id":"helpful"}`。
2. 打开“Agent 质量”，在候选策略编辑框只填写待修改字段，并写清改动假设。候选会继承当前策略未改动的字段，保存后得到新版本和父版本关系。
3. 保持“隔离 Fixture（默认）”，数据集默认指向 `tests/data/agent_eval_v1.jsonl`，点击“运行基线对比”。评测对每个案例分别创建临时 SQLite Harness，使用固定 Provider、工具和 RAG 夹具，不会调用生产 DeepSeek/MCP/API，也不会修改真实任务与记忆。
4. 若要比较编排路径，将数据集切换为 `tests/data/agent_multi_agent_v1.jsonl`，点击“运行单/多 Agent 对照”。报告验证两次回放使用相同数据集哈希，展示逐案例结果、平均调用数、平均延迟、p95 延迟及 N/A 原因；多 Agent 启用门禁要求案例通过率至少 0.80、相对单 Agent 的通过率/有标注质量指标退化不超过 0.05、越权调用为 0，并且 p95 不超过输入上限。该门禁只说明固定样例回放是否通过，不自动修改生产路由。
5. 检查策略实验报告中的数据集哈希、策略哈希、指标、N/A 原因和逐案例差异。只有所有策略门禁通过的报告可以晋级；填写该报告的 Experiment ID，勾选确认后点击“确认晋级”。
6. 若新版本表现退化，展开“回滚策略”，选择已有版本并勾选确认后执行回滚。策略历史和转换记录保留在本地 SQLite。

真实 Provider 模式单独标记为 live，需要勾选可见的外部调用确认。它会使用已配置的 DeepSeek Provider 与只读内置工具；MCP 副作用仍受原有人工确认约束。真实模式可能产生网络/API 费用，且结果具有模型非确定性。评测报告不记录 API key；fixture 报告明确标为 `fixture`，不能作为真实 Provider 的性能结论。

### 数据集与指标

Agent 回归集是小型工程回归案例，不是统计代表性 Benchmark。文件采用 UTF-8 JSONL：第一行为包含 `dataset_id`、`version`、`description` 的 manifest，后续每行为一个案例，包含任务、输入、可用工具、记忆夹具、模型/工具固定响应、期望结果与 checker ID。加载器限制文件/案例大小、案例数和 schema，并拒绝秘密值及动态代码字段。新增案例时复制现有案例结构、分配唯一 `example_id`，补齐固定响应和可核验的期望项，再运行下面的测试。

```powershell
.\.venv\Scripts\python.exe -m pytest tests\edgemind\test_agent_evaluation.py tests\edgemind\test_agent_experiments.py -q
```

- `case_pass_rate`：通过所有适用案例断言的比例；若案例要求的工具不在策略允许列表中，该案例标记为 N/A 并跳过运行，报告同时列出适用与 N/A 数量及原因。
- `allowlist_compliance` 与 `forbidden_tool_attempts`：工具调用权限合规率和越权/禁用调用计数。
- `tool_call_success_rate`：已完成工具调用中的成功比例；没有工具调用时是 N/A。
- `memory_precision_at_k` / `memory_recall_at_k`：仅在案例提供人工 gold memory ID 时计算。
- `citation_coverage`：仅在案例提供必需引用 ID 时计算。
- `expected_outcome_accuracy`：仅在案例提供期望任务结果标签时计算。
- `latency_ms_mean` / `latency_ms_p95`：当前评测运行收集到的 Agent 端到端平均耗时与最近秩 p95；多 Agent 对照门禁使用 p95。
- `model_call_count_mean`：每个案例实际 Planner/Reviewer/Executor Provider 调用数均值；候选计划评分本身不增加模型调用。硬上限为每轮 8 次。
- `token_usage_mean`：Provider 返回 usage 时统计；当前 fixture 不模拟 token usage，因此显示 N/A，不以 0 代替。
- `user_feedback_outcome_rate`：Fixture 不伪造用户标注，因此没有可链接的标注时显示 N/A。

策略候选默认要求案例通过率至少 0.8，工具成功率相对父策略退化不超过 0.05，并且越权/禁用调用为 0。报告缺少门禁所需数据、基线和候选数据集哈希不同、评测异常或存在安全失败时，晋级被阻止。该默认阈值是项目内的回归门槛，不代表业务 SLA。

新闭环规格与实施计划：`docs/specs/2026-10-10-edgemind-agent-quality-experience-loop.md`、`docs/plans/2026-10-10-edgemind-agent-quality-experience-loop.md`。

## RAG 质量评测

当前混合 RAG 适配器由旧版检索实现承载，依赖较重，作为可选 `rag` extra 安装。首次使用还需要准备私有 chunks JSON、`bge-reranker-v2-m3` 模型目录及 spaCy 中文模型 `zh_core_web_lg`；这些语料和模型不随仓库分发。该兼容适配器仍有 GPU 路径假设，CPU 端到端 RAG 行为需要结合本机环境验证。`edgemind` Harness、任务、记忆和 Fixture 评测不依赖这些本地模型。

现有 RAG 评测、反馈和配置门禁保持兼容。示例：

```powershell
python -m edgemind manifest --chunks all_pdf_page_chunks.json --parser fitz --parser-version 1
python -m edgemind query "公司收入趋势如何？" --chunks all_pdf_page_chunks.json
python -m edgemind eval --dataset benchmark.jsonl --chunks all_pdf_page_chunks.json
```

benchmark 每行至少包含 `example_id` 和 `question`，建议提供 `expected_chunk_ids`。只有存在相应 gold 标签的数据才计算对应指标。RAG 相关本地模型依赖按需加载；未配置时 Agent 的任务和记忆功能仍可使用。项目不附带私人语料、向量索引或 SQLite 状态库；请准备自己的 chunks JSON 和评测数据，相关本地文件默认被 `.gitignore` 排除。

## 隐私和安全边界

- API key 由 `.env` 或进程环境提供，不显示在页面，也不保存到 SQLite。
- Web 默认绑定 127.0.0.1，`share=False`。
- 原始图片/音频默认不持久化；数据库保存附件名、类型、大小和哈希等运行元数据。
- MCP 配置只启用用户明确开启的 server；环境变量凭证不会自动继承给子进程，需显式列入 env 映射。
- 有副作用的 MCP 工具等待用户确认；没有任意 shell、任意 Python 执行工具。
- “自进化”是反馈驱动的候选记忆、审核、离线评测和显式配置晋升，不是模型自动训练或自我改写代码。

## 文档

- 项目规格：`docs/specs/2026-10-10-edgemind-long-term-agent-harness.md`
- 实施计划：`docs/plans/2026-10-10-edgemind-agent-harness.md`
- MCP 配置样例：`docs/mcp-servers.example.json`
- 天气、网页搜索与 Skills 规格：`docs/specs/2026-10-10-edgemind-weather-web-search.md`
- 天气、网页搜索与 Skills 实施计划：`docs/plans/2026-10-10-edgemind-weather-web-search.md`
