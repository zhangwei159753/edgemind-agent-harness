import json
import os
import threading
import time
from pathlib import Path
from typing import Mapping

import gradio as gr
from dotenv import load_dotenv

from .ingestion import build_corpus_manifest, load_chunks_json
from .providers import DeepSeekSettings, create_deepseek_client
from .storage import SqliteStore

load_dotenv()
DEFAULT_CHUNKS = Path("all_pdf_page_chunks.json")
DEFAULT_STATE = Path(".data/edgemind.sqlite3")
_RAG_SERVICE = None
_RAG_LOCK = threading.Lock()
_HARNESS_SERVICE = None
_HARNESS_LOCK = threading.Lock()

CSS = """
:root { --ed-bg: #f8f9fb; --ed-sidebar: #f3f4f6; --ed-panel: #ffffff; --ed-line: #e5e7eb; --ed-text: #202123; --ed-muted: #6b7280; --ed-accent: #202123; }
.gradio-container { width: 100% !important; max-width: none !important; min-height: 100vh; margin: 0 !important; padding: 0 !important; background: var(--ed-bg) !important; color: var(--ed-text); font-family: "Segoe UI Variable", "Microsoft YaHei UI", sans-serif; }
#ed-layout { max-width: 1600px; min-height: 100vh; margin: 0 auto; gap: 0 !important; align-items: stretch; }
#ed-sidebar { position: sticky; top: 0; height: 100vh; min-width: 252px; max-width: 252px; overflow-y: auto; padding: 18px 12px; background: var(--ed-sidebar); border-right: 1px solid var(--ed-line); }
#ed-main { min-width: 0; min-height: 100vh; padding: 0 30px 28px; background: var(--ed-bg); }
#ed-brand { display:flex; gap:11px; align-items:center; padding: 8px 11px 24px; color:#202123; font-size:17px; font-weight:650; letter-spacing:-.02em; }
#ed-brand-mark { width:30px; height:30px; display:grid; place-items:center; border-radius:10px; background:#202123; color:white; font-size:13px; font-weight:700; }
.ed-nav button { min-height:40px !important; justify-content:flex-start !important; border:0 !important; border-radius:10px !important; background:transparent !important; color:#35363a !important; text-align:left !important; font-weight:500 !important; }
.ed-nav button:hover { background:#e7e8eb !important; }
#ed-task-label { margin:24px 11px 5px; color:#8b8d91; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
#ed-header { display:flex; align-items:center; justify-content:space-between; max-width:880px; margin:0 auto; padding:25px 4px 16px; border-bottom:1px solid #eceef0; }
#ed-title { color:#202123; margin:0; font-size:18px; font-weight:600; letter-spacing:-.02em; }
#ed-subtitle { margin-top:4px; color:var(--ed-muted); font-size:12px; }
.ed-page { max-width:880px; margin:0 auto; }
.ed-page-title { margin:22px 0 4px !important; font-size:25px !important; font-weight:600 !important; letter-spacing:-.04em; }
.ed-page-hint { margin:0 0 14px !important; color:var(--ed-muted); font-size:13px; }\n#ed-compose-actions { align-items:flex-end; gap:10px; }\n#ed-attachments { flex:1; border:0 !important; background:transparent !important; }\n#ed-attachments > summary { padding:8px 10px !important; color:#55585e; font-size:13px; }\n#ed-advanced, #ed-task-controls { margin:10px 0; border:1px solid var(--ed-line) !important; border-radius:12px !important; background:#fff !important; }\n.ed-audio-note { margin:4px 8px !important; color:#62666d; font-size:12px !important; line-height:1.5 !important; }\n#ed-send button { min-width:104px; min-height:42px; }\n.ed-card { background:var(--ed-panel) !important; border:1px solid var(--ed-line) !important; border-radius:14px !important; box-shadow:0 1px 2px rgba(0,0,0,.025); }
#ed-chatbot { border:0 !important; background:transparent !important; }
#ed-composer { margin-top:12px; padding:12px; border:1px solid #d9dce1; border-radius:22px; background:#fff; box-shadow:0 5px 24px rgba(18,22,33,.06); }
#ed-send button, button.primary { background:#202123 !important; color:#fff !important; border:0 !important; border-radius:12px !important; font-weight:600 !important; }
button.secondary { border:1px solid #e2e4e8 !important; border-radius:10px !important; background:#fff !important; color:#303136 !important; }
input, textarea { border-color:#e2e4e8 !important; border-radius:10px !important; }
footer { display:none !important; }
@media (max-width: 760px) { #ed-sidebar { min-width:64px; max-width:64px; padding:12px 5px; } #ed-brand { justify-content:center; padding:8px 0 20px; } #ed-brand-word,#ed-task-label,#ed-task-select,#ed-create-accordion { display:none !important; } .ed-nav button { font-size:0 !important; justify-content:center !important; } .ed-nav button:before { font-size:16px; } #ed-nav-agent button:before { content:"✦"; } #ed-nav-memory button:before { content:"◈"; } #ed-nav-tools button:before { content:"⌘"; } #ed-nav-diagnostics button:before { content:"⚙"; } #ed-main { padding:0 12px 18px; } }
"""


def run_provider_probe(prompt: str, env: Mapping[str, str] | None = None, client_factory=None) -> dict:
    """Make one explicit DeepSeek request; never return or log credential values."""
    try:
        settings = DeepSeekSettings.from_env(env)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        client = (client_factory or create_deepseek_client)(settings)
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": prompt.strip() or "请回复：DeepSeek API 连接正常。"}],
            stream=False,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        reply = response.choices[0].message.content or ""
        return {"ok": True, "model": settings.model, "latency_ms": elapsed_ms, "reply": reply}
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        failure = f"HTTP {status_code}" if status_code else type(exc).__name__
        return {"ok": False, "model": settings.model, "error": failure}


def _query_service(chunks_path: str | Path):
    global _RAG_SERVICE
    with _RAG_LOCK:
        if _RAG_SERVICE is not None:
            return _RAG_SERVICE
        chunks_path = Path(chunks_path)
        if not chunks_path.is_file():
            raise FileNotFoundError(f"chunk 文件不存在：{chunks_path}")
        chunks = load_chunks_json(chunks_path)
        manifest = build_corpus_manifest(chunks, "legacy-json", "unknown", {})
        store = SqliteStore(DEFAULT_STATE)
        store.record_corpus_manifest(manifest["corpus_version"], manifest)
        # Import only on a RAG request so provider diagnostics and the page work without NLP/model weights.
        from myrag import SimpleRAG
        from .query import QueryService

        rag = SimpleRAG(str(chunks_path))
        rag.setup()
        _RAG_SERVICE = QueryService(lambda config: rag, store, manifest["corpus_version"], "legacy-ui")
        return _RAG_SERVICE


def run_rag_query(question: str, chunks_path: str | Path = DEFAULT_CHUNKS):
    if not question or not question.strip():
        return "", {"status": "invalid", "message": "请输入问题。"}
    try:
        run = _query_service(chunks_path).ask(question.strip())
        evidence = [{"chunk_id": chunk.get("chunk_id"), "filename": (chunk.get("metadata") or {}).get("file_name"),
                     "page": (chunk.get("metadata") or {}).get("page"), "score": chunk.get("score"),
                     "content": chunk.get("content", "")[:1000]} for chunk in run.evidence]
        trace = {"run_id": run.run_id, "status": run.status, "corpus_version": run.corpus_version,
                 "config_version": run.config_version, "stage_ms": run.stage_ms, "citations": run.citations,
                 "evidence": evidence, "error_type": run.error_type}
        return run.answer, trace
    except Exception as exc:
        return "RAG 引擎未能初始化或查询失败。", {"status": "unavailable", "error_type": type(exc).__name__,
                                                   "detail": str(exc)[:300]}


def _harness_service():
    global _HARNESS_SERVICE
    with _HARNESS_LOCK:
        if _HARNESS_SERVICE is None:
            from .harness import AgentService, HarnessStore, MemoryService, TaskService
            from .harness.builtin_tools import BuiltinToolRegistry
            from .harness.policies import AgentPolicyService
            from .harness.quality import QualityService
            from .harness.tools import ToolRegistry

            store = HarnessStore(DEFAULT_STATE)
            def embed_memory(text):
                api_key = os.getenv("LOCAL_API_KEY", "").strip()
                base_url = os.getenv("LOCAL_BASE_URL", "").strip()
                model = os.getenv("LOCAL_EMBEDDING_MODEL", "").strip()
                if not (api_key and base_url and model):
                    return None
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=base_url, timeout=8.0, max_retries=0)
                response = client.embeddings.create(model=model, input=[text])
                return response.data[0].embedding

            memory = MemoryService(store, embed_query=embed_memory)
            tools = ToolRegistry(store)
            builtin_tools = BuiltinToolRegistry()

            def registered_tool_ids():
                names = {item["function"]["name"] for item in builtin_tools.model_tools()}
                try:
                    # Read only the explicit MCP tool allowlists from local config.
                    # Discovering every server here would launch stdio processes during
                    # policy lookup, including during fixture-only evaluations.
                    for server_name, config in tools.manager.configs().items():
                        configured = config.get("tools")
                        if isinstance(configured, list):
                            names.update(f"{server_name}__{tool_name}" for tool_name in configured
                                         if isinstance(tool_name, str) and tool_name.strip())
                except Exception:
                    pass
                return names

            policies = AgentPolicyService(store, registered_tool_ids=registered_tool_ids)

            class LazyRAG:
                def ask(self, text):
                    return _query_service(DEFAULT_CHUNKS).ask(text)

            _HARNESS_SERVICE = {"store": store, "memory": memory, "policies": policies,
                                "quality": QualityService(store), "tools": tools,
                                "builtin_tools": builtin_tools, "rag_service": LazyRAG(),
                                "tasks": TaskService(store),
                                "agent": AgentService(store, memory=memory, tools=tools,
                                                      builtin_tools=builtin_tools, rag_service=LazyRAG(),
                                                      policy_service=policies)}
        return _HARNESS_SERVICE


def create_harness_task(title: str, objective: str):
    try:
        task = _harness_service()["tasks"].create(title or "", objective or "")
        return task.task_id, {"status": task.status, "title": task.title, "objective": task.objective}
    except Exception as exc:
        return "", {"status": "error", "message": str(exc)[:300]}


def change_task_state(task_id: str, action: str):
    try:
        tasks = _harness_service()["tasks"]
        if action == "pause":
            task = tasks.pause(task_id)
        elif action == "resume":
            task = tasks.resume(task_id)
        elif action == "complete":
            task = tasks.complete(task_id)
        else:
            raise ValueError("unknown task action")
        return {"task_id": task.task_id, "status": task.status, "checkpoint": task.state}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:300]}





def skill_choices(registry=None):
    from .harness.skills import SkillRegistry

    registry = registry or SkillRegistry()
    return [("普通对话（自动工具）", "")] + [
        (f"{skill.title} · {skill.description}", skill.id) for skill in registry.list_skills()
    ]


def tool_readiness_status(env: Mapping[str, str] | None = None, mcp_tools=None) -> list[dict]:
    env = os.environ if env is None else env
    if mcp_tools is None:
        mcp_tools = list_mcp_tools()
    if isinstance(mcp_tools, list):
        mcp_count = len(mcp_tools)
        mcp_ready = mcp_count > 0
    else:
        mcp_count, mcp_ready = 0, False
    brave_ready = bool((env.get("BRAVE_SEARCH_API_KEY") or "").strip())
    return [
        {"id": "open_meteo", "label": "Open-Meteo 天气", "ready": True,
         "detail": "已集成；无需 API key。连通性请手动检查 · 个人非商业使用"},
        {"id": "brave_search", "label": "Brave Search", "ready": brave_ready,
         "detail": "已配置" if brave_ready else "未配置；在本机 .env 设置 BRAVE_SEARCH_API_KEY"},
        {"id": "mcp", "label": "MCP 插件", "ready": mcp_ready,
         "detail": f"已启用 {mcp_count} 个工具" if mcp_ready else "当前没有已启用的 MCP 工具"},
    ]


def check_builtin_tool_health(tool_id: str, env: Mapping[str, str] | None = None,
                              search_client=None, weather_client=None) -> dict:
    env = os.environ if env is None else env
    started = time.perf_counter()
    try:
        if tool_id == "brave_search":
            if not (env.get("BRAVE_SEARCH_API_KEY") or "").strip():
                return {"tool_id": tool_id, "ok": False, "error": "not_configured",
                        "detail": "请在本机 .env 设置 BRAVE_SEARCH_API_KEY 后重启。"}
            if search_client is None:
                from .harness.integrations.search import BraveSearchClient
                search_client = BraveSearchClient(api_key=env.get("BRAVE_SEARCH_API_KEY"), timeout_s=5)
            result = search_client.search("EdgeMind connectivity check", count=1)
            count = int(result.get("result_count", 0))
        elif tool_id == "open_meteo":
            if weather_client is None:
                from .harness.integrations.weather import OpenMeteoWeatherClient
                weather_client = OpenMeteoWeatherClient(timeout_s=5)
            result = weather_client.get_weather("北京")
            response_status = result.get("status", "unavailable")
            count = 1 if response_status == "succeeded" else 0
            if response_status in {"needs_clarification", "not_found"}:
                return {"tool_id": tool_id, "ok": True, "result_count": 0,
                        "response_status": response_status,
                        "detail": "天气服务已响应；检查地点存在多个候选，实际查询时请补充省份或国家。",
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
            if not count:
                return {"tool_id": tool_id, "ok": False, "error": response_status,
                        "detail": str(result.get("message", "天气服务未返回可用结果"))[:180]}
        else:
            return {"tool_id": tool_id, "ok": False, "error": "unknown_tool",
                    "detail": "请选择 Open-Meteo 天气或 Brave Search。"}
        return {"tool_id": tool_id, "ok": True, "result_count": count,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
    except Exception as exc:
        return {"tool_id": tool_id, "ok": False,
                "error": getattr(exc, "category", type(exc).__name__),
                "detail": "连通检查失败，请检查配置、网络或稍后重试。",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def format_health_result(result: dict) -> str:
    label = "正常" if result.get("ok") else "未通过"
    detail = result.get("detail") or (
        f"{result.get('result_count', 0)} 条结果 · {result.get('latency_ms', 0)} ms"
        if result.get("ok") else f"错误类别：{result.get('error', 'unknown')}"
    )
    return f"**{label}**　{detail}"


def format_answer_with_sources(answer: str, trace: dict) -> str:
    import re
    from urllib.parse import urlparse

    links, seen = [], set()
    for call in trace.get("tool_calls", []):
        for url in call.get("source_urls", []):
            parsed = urlparse(url if isinstance(url, str) else "")
            if parsed.scheme in {"http", "https"} and parsed.hostname and url not in seen:
                seen.add(url)
                safe_host = re.sub(r"[^A-Za-z0-9_.:-]", "", parsed.hostname)
                safe_url = (url.replace("\\", "%5C").replace("(", "%28")
                            .replace(")", "%29").replace(" ", "%20"))
                links.append(f"- [{safe_host}]({safe_url})")
    return answer if not links else answer.rstrip() + "\n\n**来源**\n" + "\n".join(links)


def run_agent_turn(task_id: str, text: str, images, audio, skill_id: str = "", progress=None,
                   orchestration_mode: str = "auto"):
    if not task_id or not task_id.strip():
        return "请先创建任务并填写任务 ID。", {"status": "invalid"}, ""
    try:
        turn = _harness_service()["agent"].run_turn(
            task_id.strip(), text or "", images or [], audio or None, skill_id=skill_id or None,
            orchestration_mode=orchestration_mode or "auto",
            on_tool_status=(lambda description: progress(desc=description)) if progress else None)
        return turn.answer, {**turn.trace, "pending_tool_calls": turn.pending_tool_calls}, turn.run_id
    except Exception as exc:
        return f"本轮未完成：{type(exc).__name__}", {"status": "error", "detail": str(exc)[:300]}, ""


def record_agent_feedback(task_id: str, run_id: str, outcome: str, failure_category: str,
                          memory_feedback_json: str, note: str):
    try:
        raw = (memory_feedback_json or "{}").strip() or "{}"
        memory_feedback = json.loads(raw)
        if not isinstance(memory_feedback, dict):
            raise ValueError("记忆反馈必须是 JSON 对象")
        result = _harness_service()["quality"].record_feedback(
            task_id.strip(), run_id.strip(), outcome, failure_category or None,
            memory_feedback=memory_feedback, note=note or "")
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


def agent_policy_choices():
    try:
        policies = _harness_service()["policies"].list_policies()
        return [(f"v{item['version']} · {item['policy_id'][:12]}{' · 当前' if item['active'] else ''}",
                 item["policy_id"]) for item in policies]
    except Exception:
        return [("内置基线", "baseline-v1")]


def agent_policy_dropdown_updates(select_policy_id: str | None = None):
    try:
        items = _harness_service()["policies"].list_policies()
        choices = [(f"v{item['version']} · {item['policy_id'][:12]}{' · 当前' if item['active'] else ''}",
                    item["policy_id"]) for item in items]
        valid_ids = {value for _, value in choices}
        active_id = next((item["policy_id"] for item in items if item["active"]), "baseline-v1")
        selected = select_policy_id if select_policy_id in valid_ids else active_id
        return (gr.update(choices=choices, value=selected),
                gr.update(choices=choices, value=selected))
    except Exception:
        fallback = [("內置基线", "baseline-v1")]
        return gr.update(choices=fallback, value="baseline-v1"), gr.update(choices=fallback, value="baseline-v1")


def create_agent_policy_candidate(payload_json: str, rationale: str, source_run_id: str = ""):
    try:
        payload = json.loads(payload_json or "{}")
        policy = _harness_service()["policies"].create_candidate(
            payload, rationale, source_run_id.strip() or None)
        return {"ok": True, "policy": {"policy_id": policy.policy_id, "version": policy.version,
                                        "parent_policy_id": policy.parent_policy_id,
                                        "prompt_suffix": policy.prompt_suffix,
                                        "tool_allowlist": list(policy.tool_allowlist),
                                        "memory_top_k": policy.memory_top_k}}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


def run_agent_experiment(policy_id: str, dataset_path: str, mode: str, repetitions: int,
                         confirmed_live: bool):
    try:
        from .agent_eval_fixtures import build_evaluation_runner
        from .agent_experiments import AgentExperimentService, AgentQualityGates

        services = _harness_service()
        if not dataset_path or not Path(dataset_path).is_file():
            raise ValueError("评测数据集文件不存在")
        runner = build_evaluation_runner(services)
        experiments = AgentExperimentService(services["policies"], runner, services["store"],
                                             AgentQualityGates(
                                                 required_min={"case_pass_rate": 0.8},
                                                 max_regression={"tool_call_success_rate": 0.05}))
        return experiments.evaluate_candidate(policy_id, dataset_path, mode=mode,
                                              repetitions=int(repetitions), confirmed_live=bool(confirmed_live))
    except Exception as exc:
        return {"passed": False, "error_category": type(exc).__name__, "error_message": str(exc)[:500]}


def promote_agent_experiment(experiment_id: str, confirmed: bool):
    try:
        from .agent_experiments import AgentExperimentService, AgentQualityGates
        services = _harness_service()
        experiments = AgentExperimentService(services["policies"], None, services["store"], AgentQualityGates())
        policy = experiments.promote(experiment_id.strip(), confirmed=bool(confirmed))
        return {"ok": True, "active_policy_id": policy.policy_id, "version": policy.version}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


def rollback_agent_policy(policy_id: str, confirmed: bool):
    try:
        from .agent_experiments import AgentExperimentService, AgentQualityGates
        services = _harness_service()
        experiments = AgentExperimentService(services["policies"], None, services["store"], AgentQualityGates())
        policy = experiments.rollback(policy_id.strip(), confirmed=bool(confirmed))
        return {"ok": True, "active_policy_id": policy.policy_id, "version": policy.version}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


def apply_agent_correction(task_id: str, run_id: str, correction: str):
    try:
        turn = _harness_service()["agent"].correct_turn(task_id.strip(), run_id.strip(), correction)
        return turn.answer, turn.trace, turn.run_id
    except Exception as exc:
        return "纠错处理未完成。", {"status": "error", "message": str(exc)[:300]}, run_id


def list_harness_memories(status: str = "pending"):
    try:
        items = _harness_service()["store"].list_memories(status=status, limit=100)
        return [{"memory_id": item.memory_id, "layer": item.layer, "content": item.content,
                 "source_task_id": item.source_task_id, "source_run_id": item.source_run_id,
                 "confidence": item.confidence, "status": item.status} for item in items]
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:300]}


def review_harness_memory(memory_id: str, decision: str, content: str = ""):
    try:
        item = _harness_service()["memory"].review(memory_id.strip(), decision, content or None)
        return {"memory_id": item.memory_id, "status": item.status, "content": item.content}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:300]}


def propose_harness_memory(layer: str, content: str, task_id: str = ""):
    try:
        item = _harness_service()["memory"].propose(layer, content, task_id.strip() or None,
                                                   metadata={"created_from": "web_review_panel"})
        return {"memory_id": item.memory_id, "layer": item.layer, "status": item.status,
                "content": item.content, "message": "已创建待审核候选"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:300]}


def delete_harness_memory(memory_id: str):
    try:
        deleted = _harness_service()["memory"].delete(memory_id.strip())
        return {"memory_id": memory_id.strip(), "deleted": deleted}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:300]}


def list_mcp_tools():
    try:
        from .harness.tools import ToolRegistry

        service = _harness_service()
        return ToolRegistry(service["store"]).list_enabled()
    except Exception as exc:
        return {"status": "unavailable", "message": str(exc)[:300]}


def execute_mcp_proposal(proposal_id: str, confirmed: bool):
    try:
        result = _harness_service()["agent"].execute_tool_proposal(proposal_id.strip(), confirmed)
        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:300]}


def reject_mcp_proposal(proposal_id: str):
    try:
        return _harness_service()["agent"].reject_tool_proposal(proposal_id.strip())
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:300]}


def confirm_mcp_tool(proposal_id: str, history):
    result = execute_mcp_proposal(proposal_id, True)
    history = list(history or [])
    if result.get("final_answer"):
        history.append({"role": "assistant", "content": result["final_answer"]})
    return result, history


def task_choices():
    try:
        tasks = _harness_service()["store"].list_tasks(limit=100)
        return [(f"{task.title} · {task.status}", task.task_id) for task in tasks]
    except Exception:
        return []


def load_task_view(task_id: str):
    if not task_id:
        return [], {"status": "no_task"}
    try:
        service = _harness_service()
        task = service["store"].get_task(task_id)
        if task is None:
            return [], {"status": "missing", "task_id": task_id}
        history = []
        for event in reversed(service["store"].list_events(task_id, limit=300)):
            if event.event_type == "agent_turn":
                if event.payload.get("user_text"):
                    history.append({"role": "user", "content": event.payload["user_text"]})
                history.append({"role": "assistant", "content": event.payload.get("answer", "")})
            elif event.event_type == "agent_tool_followup" and event.payload.get("answer"):
                history.append({"role": "assistant", "content": event.payload["answer"]})
        return history, {"task_id": task.task_id, "status": task.status, "checkpoint": task.state}
    except Exception as exc:
        return [], {"status": "error", "message": str(exc)[:300]}


def create_task_ui(title: str, objective: str):
    task_id, status = create_harness_task(title, objective)
    return gr.Dropdown(choices=task_choices(), value=task_id or None), status, [], ""


def load_task_ui(task_id: str):
    return load_task_view(task_id)


def agent_chat_turn(task_id: str, text: str, images, audio, history, skill_id: str = "",
                    orchestration_mode: str = "auto", progress=gr.Progress(track_tqdm=False)):
    answer, trace, run_id = run_agent_turn(task_id, text, images, audio, skill_id, progress,
                                           orchestration_mode)
    answer = format_answer_with_sources(answer, trace)
    failed = trace.get("status") in {"failed", "error", "invalid"}
    history = list(history or [])
    user_label = (text or "").strip()
    if images:
        user_label = (user_label + "\n[图片附件]").strip()
    if audio:
        user_label = (user_label + "\n[音频附件]").strip()
    if user_label:
        history.append({"role": "user", "content": user_label})
    history.append({"role": "assistant", "content": answer})
    return history, trace, run_id, (text or "") if failed else "", images if failed else None, audio if failed else None


def audio_backend_status() -> str:
    import importlib.util

    model = os.getenv("EDGEMIND_WHISPER_MODEL", "small")
    if importlib.util.find_spec("faster_whisper") is None:
        return ("语音 ASR 未就绪：本地 faster-whisper 依赖未安装。"
                "请在项目环境运行 `uv sync --extra audio` 并重启；音频不会发送给 DeepSeek。")
    device = os.getenv("EDGEMIND_ASR_DEVICE", "cpu")
    compute_type = os.getenv("EDGEMIND_ASR_COMPUTE_TYPE", "int8")
    return (f"语音 ASR 已就绪：本地 Whisper {model} · {device}/{compute_type}。"
            "首次使用可能需要下载并加载模型；仅转写文本会发送给 DeepSeek。")


def correct_chat_turn(task_id: str, run_id: str, correction: str, history):
    answer, trace, new_run_id = apply_agent_correction(task_id, run_id, correction)
    history = list(history or [])
    history.extend([{"role": "user", "content": f"纠错：{correction}"}, {"role": "assistant", "content": answer}])
    return history, trace, new_run_id, ""


def build_demo():
    with gr.Blocks(title="EdgeMind · Personal Agent") as demo:
        with gr.Row(elem_id="ed-layout"):
            with gr.Column(scale=0, min_width=252, elem_id="ed-sidebar"):
                gr.HTML('<div id="ed-brand"><span id="ed-brand-mark">E</span><span id="ed-brand-word">EdgeMind</span></div>')
                agent_nav = gr.Button("✦　Agent 工作台", elem_id="ed-nav-agent", elem_classes=["ed-nav"])
                memory_nav = gr.Button("◈　记忆管理", elem_id="ed-nav-memory", elem_classes=["ed-nav"])
                tools_nav = gr.Button("⌘　Agent 工具", elem_id="ed-nav-tools", elem_classes=["ed-nav"])
                quality_nav = gr.Button("◎　Agent 质量", elem_id="ed-nav-quality", elem_classes=["ed-nav"])
                diagnostic_nav = gr.Button("⚙　诊断与评测", elem_id="ed-nav-diagnostics", elem_classes=["ed-nav"])
                gr.HTML('<div id="ed-task-label">持续任务</div>')
                task_selector = gr.Dropdown(label="选择任务", choices=task_choices(), value=None, elem_id="ed-task-select", show_label=False)
                with gr.Accordion("＋ 新建任务", open=False, elem_id="ed-create-accordion"):
                    task_title = gr.Textbox(label="任务名称", placeholder="例如：完成 Agent MVP")
                    task_objective = gr.Textbox(label="目标与完成条件", lines=3)
                    create_task = gr.Button("创建任务", variant="primary")
                gr.Markdown("本地运行 · 数据存于 SQLite")

            with gr.Column(scale=1, elem_id="ed-main"):
                gr.HTML('<header id="ed-header"><div><h1 id="ed-title">EdgeMind</h1><div id="ed-subtitle">长期任务与记忆 Harness</div></div><div>LOCAL · PRIVATE</div></header>')
                with gr.Column(visible=True, elem_classes=["ed-page"], elem_id="ed-page-agent") as page_agent:
                    gr.Markdown("### 继续推进你的任务", elem_classes=["ed-page-title"])
                    gr.Markdown("从左侧选择一项任务，或直接创建新的长期任务。", elem_classes=["ed-page-hint"])
                    with gr.Accordion("任务与进度", open=False, elem_id="ed-task-controls"):
                        task_status = gr.JSON(label="当前任务", elem_classes=["ed-card"])
                        with gr.Row():
                            pause = gr.Button("暂停", elem_classes=["secondary"])
                            resume = gr.Button("恢复", elem_classes=["secondary"])
                            complete = gr.Button("完成任务", elem_classes=["secondary"])
                    chatbot = gr.Chatbot(label="对话", height=520, elem_id="ed-chatbot", buttons=["copy"],
                                         placeholder="创建或选择任务后，在这里描述目标、进展和下一步。")
                    with gr.Group(elem_id="ed-composer"):
                        user_text = gr.Textbox(label="发送消息", placeholder="给 EdgeMind 发消息…", lines=2, show_label=False)
                        skill_selector = gr.Dropdown(label="工作流", choices=skill_choices(), value="",
                                                     info="工作流会限制可用工具；普通对话自动选择已启用工具。")
                        orchestration_selector = gr.Dropdown(
                            label="Agent 协作模式",
                            choices=[("自动（默认）", "auto"), ("单 Agent", "single"), ("多 Agent 协作", "multi")],
                            value="auto",
                            info="自动模式按任务复杂度路由；多 Agent 会增加规划与审核调用。",
                        )
                        with gr.Row(elem_id="ed-compose-actions"):
                            with gr.Accordion("添加图片与语音", open=False, elem_id="ed-attachments"):
                                gr.Markdown("图片会发送给 DeepSeek；音频在本机转写，只有文字进入对话。首次使用可能较慢。", elem_classes=["ed-audio-note"])
                                gr.Markdown(audio_backend_status(), elem_classes=["ed-audio-note"])
                                with gr.Row():
                                    images = gr.File(label="图片", file_count="multiple", file_types=["image"], type="filepath", scale=1)
                                    audio = gr.Audio(label="语音", sources=["upload", "microphone"], type="filepath", scale=1)
                            send = gr.Button("发送 ↑", variant="primary", elem_id="ed-send", scale=0)
                    with gr.Accordion("纠错与运行详情", open=False, elem_id="ed-advanced"):
                        correction = gr.Textbox(label="纠正或补充", placeholder="指出需要修正的事实或补充的信息…", lines=2)
                        apply_correction = gr.Button("提交纠错并重答", elem_classes=["secondary"])
                        last_run_id = gr.Textbox(label="当前 Run ID", interactive=False)
                        turn_trace = gr.JSON(label="运行轨迹 · Agent 路由、角色协作、证据与工具提案")
                    with gr.Accordion("结果反馈", open=False, elem_id="ed-feedback"):
                        gr.Markdown("反馈会关联到当前 Run，并用于评估 Agent 质量；不会自动修改记忆或策略。")
                        feedback_outcome = gr.Radio(label="本轮结果", choices=["success", "partial", "failure"], value="success")
                        feedback_category = gr.Dropdown(label="主要问题（失败或部分完成时）",
                                                        choices=["planning", "memory", "retrieval/evidence", "tool selection",
                                                                 "tool execution", "answer quality", "provider/network",
                                                                 "user input", "other"], value=None)
                        feedback_memory = gr.Textbox(label="记忆反馈 JSON（可选）", value="{}",
                                                     placeholder='{"memory-id":"helpful" 或 "not_helpful"}')
                        feedback_note = gr.Textbox(label="补充说明（可选）", lines=2, max_lines=4)
                        submit_feedback = gr.Button("记录本轮反馈", elem_classes=["secondary"])
                        feedback_result = gr.JSON(label="反馈记录")
                with gr.Column(visible=False, elem_classes=["ed-page"], elem_id="ed-page-memory") as page_memory:
                    gr.Markdown("### 记忆管理", elem_classes=["ed-page-title"])
                    gr.Markdown("记忆按层分类。新候选先审核，批准后才参与 Agent 上下文。")
                    with gr.Group(elem_classes=["ed-card"]):
                        with gr.Row():
                            new_memory_layer = gr.Dropdown(label="记忆层", choices=["semantic", "procedural", "episodic"], value="semantic")
                            new_memory_task = gr.Textbox(label="来源任务 ID（可选）")
                        new_memory_content = gr.Textbox(label="新增事实、偏好或经验", lines=3)
                        propose_memory = gr.Button("提交记忆候选", variant="primary")
                        proposed_memory_result = gr.JSON(label="候选创建结果")
                    memory_status = gr.Dropdown(label="查看状态", choices=["pending", "approved", "rejected"], value="pending")
                    refresh_memories = gr.Button("刷新列表", elem_classes=["secondary"])
                    memory_list = gr.JSON(label="记忆条目", elem_classes=["ed-card"])
                    memory_id = gr.Textbox(label="记忆 ID")
                    memory_edit = gr.Textbox(label="编辑内容（留空保持原文）", lines=3)
                    with gr.Row():
                        approve_memory = gr.Button("批准", variant="primary")
                        reject_memory = gr.Button("拒绝", elem_classes=["secondary"])
                        delete_memory = gr.Button("删除", elem_classes=["secondary"])
                    memory_result = gr.JSON(label="审核结果")

                with gr.Column(visible=False, elem_classes=["ed-page"], elem_id="ed-page-tools") as page_tools:
                    gr.Markdown("### Agent 工具", elem_classes=["ed-page-title"])
                    gr.Markdown("查看天气、网页搜索与 MCP 的可用状态。内置查询为只读；MCP 有副作用的调用仍需人工确认。")
                    refresh_tool_status = gr.Button("刷新工具状态", variant="primary")
                    builtin_status = gr.JSON(label="内置工具与连接状态", value=tool_readiness_status(), elem_classes=["ed-card"])
                    with gr.Row():
                        health_tool = gr.Dropdown(label="连通性检查", choices=[("Open-Meteo 天气", "open_meteo"), ("Brave Search", "brave_search")], value="open_meteo")
                        health_check = gr.Button("执行一次检查", elem_classes=["secondary"])
                    health_result = gr.Markdown("连通性检查仅在点击后发送一次有界请求。")
                    discover_tools = gr.Button("发现已启用 MCP 工具", elem_classes=["secondary"])
                    mcp_tools = gr.JSON(label="MCP 工具清单", elem_classes=["ed-card"])
                    proposal_id = gr.Textbox(label="Agent 运行轨迹中的 proposal ID")
                    with gr.Row():
                        confirm_tool = gr.Button("确认并执行", variant="primary")
                        deny_tool = gr.Button("拒绝调用", elem_classes=["secondary"])
                    tool_result = gr.JSON(label="工具执行结果")
                    gr.Markdown("配置样例位于 docs/mcp-servers.example.json；复制为 mcp_servers.json 后再启用。")

                with gr.Column(visible=False, elem_classes=["ed-page"], elem_id="ed-page-quality") as page_quality:
                    gr.Markdown("### Agent 质量与经验闭环", elem_classes=["ed-page-title"])
                    gr.Markdown("记录真实运行反馈，创建有版本的策略候选，在同一数据集上对比基线；只有通过质量门禁并人工确认后才可晋级。",
                                 elem_classes=["ed-page-hint"])
                    with gr.Group(elem_classes=["ed-card"]):
                        gr.Markdown("#### 当前策略与历史版本")
                        refresh_policies = gr.Button("刷新策略列表", elem_classes=["secondary"])
                        policy_list = gr.JSON(label="策略版本")
                    with gr.Accordion("创建策略候选", open=True):
                        gr.Markdown("只填写需要变化的字段；支持 prompt_suffix、memory_top_k、layer_weights、tag 过滤、tool_allowlist、max_tool_rounds、output_constraints 和 checker_ids。")
                        policy_payload = gr.Code(label="候选策略 JSON", language="json", value='{"prompt_suffix":"优先核对证据；信息不足时明确说明。"}')
                        policy_rationale = gr.Textbox(label="改动假设", placeholder="说明该策略要改善哪项质量问题，以及预期影响。")
                        policy_source_run = gr.Textbox(label="来源 Run ID（可选）")
                        create_policy = gr.Button("保存候选版本", variant="primary")
                        policy_create_result = gr.JSON(label="候选版本")
                    with gr.Group(elem_classes=["ed-card"]):
                        gr.Markdown("#### 候选评测与晋级")
                        quality_policy = gr.Dropdown(label="候选策略", choices=agent_policy_choices(), value=None)
                        eval_dataset = gr.Textbox(label="评测数据集 JSONL", value=str(Path(__file__).resolve().parents[1] / "tests" / "data" / "agent_eval_v1.jsonl"))
                        with gr.Row():
                            eval_mode = gr.Radio(label="运行模式", choices=[("隔离 Fixture（默认）", "fixture"), ("真实 Provider", "live")], value="fixture")
                            eval_repetitions = gr.Number(label="重复次数", value=1, minimum=1, maximum=5, precision=0)
                        live_confirm = gr.Checkbox(label="我确认真实评测会调用已配置 Provider 与只读工具", value=False)
                        evaluate_policy = gr.Button("运行基线对比", variant="primary")
                        experiment_result = gr.JSON(label="门禁报告与逐案例差异")
                        experiment_id = gr.Textbox(label="晋级所需 Experiment ID")
                        promote_confirm = gr.Checkbox(label="我确认将通过门禁的候选设为当前策略", value=False)
                        promote_policy = gr.Button("确认晋级", elem_classes=["secondary"])
                        promote_result = gr.JSON(label="晋级结果")
                    with gr.Accordion("回滚策略", open=False):
                        rollback_policy = gr.Dropdown(label="恢复到策略版本", choices=agent_policy_choices(), value="baseline-v1")
                        rollback_confirm = gr.Checkbox(label="我确认切换当前策略", value=False)
                        rollback_button = gr.Button("确认回滚", elem_classes=["secondary"])
                        rollback_result = gr.JSON(label="回滚结果")

                with gr.Column(visible=False, elem_classes=["ed-page"], elem_id="ed-page-diagnostics") as page_diagnostics:
                    gr.Markdown("### 诊断与评测", elem_classes=["ed-page-title"])
                    with gr.Accordion("DeepSeek API 连通性", open=True):
                        gr.Markdown("只在点击按钮后发送一次请求。密钥从本机 .env 读取，不显示在页面。")
                        prompt = gr.Textbox(label="测试提示词", value="请用一句话回复：EdgeMind API 连通正常。", lines=2)
                        probe = gr.Button("发送一次测试请求", variant="primary")
                        api_result = gr.JSON(label="连接结果")
                        probe.click(lambda text: run_provider_probe(text), inputs=prompt, outputs=api_result)
                    with gr.Accordion("RAG 查询轨迹", open=False):
                        gr.Markdown("RAG 本地模型按需加载。缺少模型时不影响任务与记忆页面启动。")
                        question = gr.Textbox(label="问题", placeholder="输入一个知识库问题", lines=2)
                        chunks = gr.Textbox(label="Chunk JSON", value=str(DEFAULT_CHUNKS))
                        ask = gr.Button("运行检索与生成", variant="primary")
                        rag_answer = gr.Textbox(label="回答", lines=5, interactive=False)
                        trace = gr.JSON(label="运行轨迹与证据")
                        ask.click(run_rag_query, inputs=[question, chunks], outputs=[rag_answer, trace])
                    gr.Markdown("本地地址默认绑定 127.0.0.1。运行数据保存在 .data/edgemind.sqlite3。")

        pages = [page_agent, page_memory, page_tools, page_quality, page_diagnostics]
        def navigate(index):
            return [gr.update(visible=position == index) for position in range(len(pages))]
        agent_nav.click(lambda: navigate(0), outputs=pages)
        memory_nav.click(lambda: navigate(1), outputs=pages)
        tools_nav.click(lambda: navigate(2), outputs=pages)
        quality_nav.click(lambda: navigate(3), outputs=pages)
        diagnostic_nav.click(lambda: navigate(4), outputs=pages)

        create_task.click(create_task_ui, inputs=[task_title, task_objective], outputs=[task_selector, task_status, chatbot, task_title])
        task_selector.change(load_task_ui,
                             inputs=task_selector, outputs=[chatbot, task_status])
        pause.click(lambda identity: change_task_state(identity, "pause"), inputs=task_selector, outputs=task_status)
        resume.click(lambda identity: change_task_state(identity, "resume"), inputs=task_selector, outputs=task_status)
        complete.click(lambda identity: change_task_state(identity, "complete"), inputs=task_selector, outputs=task_status)
        send.click(agent_chat_turn, inputs=[task_selector, user_text, images, audio, chatbot, skill_selector, orchestration_selector], outputs=[chatbot, turn_trace, last_run_id, user_text, images, audio])
        user_text.submit(agent_chat_turn, inputs=[task_selector, user_text, images, audio, chatbot, skill_selector, orchestration_selector], outputs=[chatbot, turn_trace, last_run_id, user_text, images, audio])
        apply_correction.click(correct_chat_turn, inputs=[task_selector, last_run_id, correction, chatbot], outputs=[chatbot, turn_trace, last_run_id, correction])
        submit_feedback.click(record_agent_feedback,
                              inputs=[task_selector, last_run_id, feedback_outcome, feedback_category, feedback_memory, feedback_note],
                              outputs=feedback_result)
        propose_memory.click(propose_harness_memory, inputs=[new_memory_layer, new_memory_content, new_memory_task], outputs=proposed_memory_result)
        refresh_memories.click(list_harness_memories, inputs=memory_status, outputs=memory_list)
        memory_status.change(list_harness_memories, inputs=memory_status, outputs=memory_list)
        approve_memory.click(lambda identity, content: review_harness_memory(identity, "approved", content), inputs=[memory_id, memory_edit], outputs=memory_result)
        reject_memory.click(lambda identity, content: review_harness_memory(identity, "rejected", content), inputs=[memory_id, memory_edit], outputs=memory_result)
        delete_memory.click(delete_harness_memory, inputs=memory_id, outputs=memory_result)
        refresh_tool_status.click(lambda: tool_readiness_status(), outputs=builtin_status)
        discover_tools.click(list_mcp_tools, outputs=mcp_tools)
        health_check.click(lambda identity: format_health_result(check_builtin_tool_health(identity)), inputs=health_tool, outputs=health_result)
        confirm_tool.click(confirm_mcp_tool, inputs=[proposal_id, chatbot], outputs=[tool_result, chatbot])
        deny_tool.click(reject_mcp_proposal, inputs=proposal_id, outputs=tool_result)
        refresh_policies.click(lambda: _harness_service()["policies"].list_policies(), outputs=policy_list).then(
            agent_policy_dropdown_updates, outputs=[quality_policy, rollback_policy])
        create_policy.click(create_agent_policy_candidate,
                            inputs=[policy_payload, policy_rationale, policy_source_run], outputs=policy_create_result).then(
            lambda result: agent_policy_dropdown_updates(
                result.get("policy", {}).get("policy_id") if result.get("ok") else None),
            inputs=policy_create_result, outputs=[quality_policy, rollback_policy])
        evaluate_policy.click(run_agent_experiment,
                              inputs=[quality_policy, eval_dataset, eval_mode, eval_repetitions, live_confirm],
                              outputs=experiment_result)
        promote_policy.click(promote_agent_experiment, inputs=[experiment_id, promote_confirm], outputs=promote_result).then(
            agent_policy_dropdown_updates, outputs=[quality_policy, rollback_policy])
        rollback_button.click(rollback_agent_policy, inputs=[rollback_policy, rollback_confirm], outputs=rollback_result).then(
            agent_policy_dropdown_updates, outputs=[quality_policy, rollback_policy])
    return demo


def main():
    port = int(os.getenv("EDGEMIND_PORT", "7860"))
    build_demo().queue().launch(server_name="127.0.0.1", server_port=port, share=False, css=CSS)


if __name__ == "__main__":
    main()
