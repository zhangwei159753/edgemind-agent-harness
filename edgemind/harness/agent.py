import json
import hashlib
import os
import re
import time
import uuid
from typing import Any, Callable

from ..providers import DeepSeekSettings, create_deepseek_client
from .media import MediaProcessor
from .memory import MemoryService
from .models import AgentTurn, MemoryItem
from .store import HarnessStore
from .tasks import TaskService
from .tools import ToolRegistry
from .builtin_tools import BuiltinToolRegistry
from .policies import AgentPolicy, AgentPolicyService
from .skills import SkillRegistry, SkillDefinition


class AgentService:
    def __init__(self, store: HarnessStore, client=None, media: MediaProcessor | None = None,
                 memory: MemoryService | None = None, tools: ToolRegistry | None = None,
                 rag_service=None, builtin_tools: BuiltinToolRegistry | None = None,
                 policy_service: AgentPolicyService | None = None):
        self.store = store
        self.client = client
        self.media = media or MediaProcessor()
        self.memory = memory or MemoryService(store)
        self.tasks = TaskService(store)
        self.tools = tools or ToolRegistry(store)
        self.builtin_tools = builtin_tools or BuiltinToolRegistry()
        self.skills = SkillRegistry()
        self.rag_service = rag_service
        self.policy_service = policy_service

    def _client(self):
        if self.client is None:
            self.client = create_deepseek_client(DeepSeekSettings.from_env())
        return self.client


    @staticmethod
    def _parse_tool_arguments(raw: Any) -> dict:
        if isinstance(raw, dict):
            arguments = raw
        elif isinstance(raw, str):
            if len(raw) > 50000:
                raise ValueError("tool arguments exceed 50 KiB")
            try:
                arguments = json.loads(raw or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("tool arguments are not valid JSON") from exc
        else:
            raise ValueError("tool arguments must be a JSON object")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        return arguments

    @staticmethod
    def _tool_result_content(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)[:16000]

    @staticmethod
    def _safe_tool_context(messages: list[dict]) -> list[dict]:
        safe = []
        for message in messages[1:]:
            item = dict(message)
            content = item.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": str(part.get("text", ""))[:20000]})
                    elif part.get("type") in {"image_url", "input_image"}:
                        parts.append({"type": "text", "text": "[图片附件已从确认续答上下文中省略]"})
                item["content"] = parts
            elif isinstance(content, str):
                item["content"] = content[:16000]
            safe.append(item)
        serialized = json.dumps(safe, ensure_ascii=False, default=str)
        if len(serialized) <= 60000:
            return safe
        # Keep the original user request and the most recent tool context; never persist media bytes.
        compact = []
        for item in safe:
            if item.get("role") == "user" and not compact:
                compact.append(item)
        compact.extend([item for item in safe if item.get("role") in {"assistant", "tool"}][-8:])
        return compact

    def _run_tool_loop(self, messages: list[dict], task_id: str, run_id: str,
                       stage_ms: dict, user_text: str,
                       on_tool_status: Callable[[str], None] | None = None,
                       skill: SkillDefinition | None = None, policy: AgentPolicy | None = None,
                       trace_collector: list[dict] | None = None):
        tool_warning = None
        try:
            mcp_tools = self.tools.model_tools()
        except Exception as exc:
            mcp_tools = []
            tool_warning = type(exc).__name__
        try:
            builtin_tools = self.builtin_tools.model_tools()
        except Exception as exc:
            builtin_tools = []
            tool_warning = tool_warning or type(exc).__name__
        model_tools = builtin_tools + mcp_tools
        if policy is not None:
            model_tools = [item for item in model_tools
                           if item.get("function", {}).get("name") in policy.tool_allowlist]
        if skill is not None:
            model_tools = [item for item in model_tools
                           if item.get("function", {}).get("name") in skill.allowed_tools]
        enabled_tool_names = {item.get("function", {}).get("name") for item in model_tools}
        max_rounds = min(policy.max_tool_rounds if policy else 3,
                         skill.max_tool_rounds if skill else 3)
        fetched_pages = 0
        allowed_urls = {
            item.rstrip(".,;:!?)]}") for item in re.findall(r"https?://[^\s<>]+", user_text or "")
        }
        tool_trace = trace_collector if trace_collector is not None else []
        proposals = []
        proposal_calls = []
        tool_call_data = None
        rounds = 0
        force_final = False

        while True:
            kwargs = {
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                "messages": messages,
                "stream": False,
            }
            if model_tools and not force_final:
                kwargs.update({"tools": model_tools, "tool_choice": "auto"})
            model_started = time.perf_counter()
            response = self._client().chat.completions.create(**kwargs)
            stage_ms[f"deepseek_round_{rounds + 1}"] = round(
                (time.perf_counter() - model_started) * 1000, 2
            )
            message = response.choices[0].message
            calls = list(getattr(message, "tool_calls", None) or [])
            if not calls:
                return (message.content or "", "succeeded", proposals, tool_call_data,
                        tool_trace, rounds, tool_warning, None, proposal_calls)
            if force_final:
                return (message.content or "已达到工具调用轮次上限，以上信息基于已取得的结果。",
                        "succeeded", proposals, tool_call_data, tool_trace, rounds,
                        tool_warning, None, proposal_calls)

            rounds += 1
            assistant_calls = []
            for call in calls:
                function = call.function
                call_id = getattr(call, "id", None) or f"{run_id}-{len(assistant_calls)}"
                item = {
                    "id": call_id, "type": "function",
                    "function": {"name": function.name, "arguments": function.arguments},
                }
                assistant_calls.append(item)
                if tool_call_data is None:
                    tool_call_data = item
            messages.append({"role": "assistant", "content": message.content or None,
                             "tool_calls": assistant_calls})
            round_results = []

            for item in assistant_calls:
                function = item["function"]
                name = function["name"]
                if on_tool_status:
                    try:
                        label = name.split("__")[-1]
                        on_tool_status(f"正在调用 {label}")
                    except Exception:
                        pass
                tool_started = time.perf_counter()
                error_type = None
                result = None
                proposal = None
                try:
                    arguments = self._parse_tool_arguments(function.get("arguments"))
                    if name not in enabled_tool_names:
                        error_type = "tool_not_allowed"
                        result = {"status": "error", "error_type": error_type,
                                  "message": "该工作流未授权此工具。"}
                    elif name == "edgemind__fetch_search_result" and skill is not None and fetched_pages >= 3:
                        error_type = "skill_limit"
                        result = {"status": "error", "error_type": error_type,
                                  "message": "该工作流最多读取三个网页。"}
                    elif name.startswith("edgemind__"):
                        if name == "edgemind__fetch_search_result" and skill is not None:
                            fetched_pages += 1
                        result = self.builtin_tools.execute(
                            name, arguments, allowed_urls=allowed_urls
                        )
                        if name == "edgemind__web_search" and isinstance(result, dict):
                            for source in result.get("results", []):
                                url = source.get("url")
                                if isinstance(url, str):
                                    allowed_urls.add(url)
                    else:
                        proposal = self.tools.propose_call(
                            task_id, run_id, name, json.dumps(arguments, ensure_ascii=False)
                        )
                        proposal["tool_call_id"] = item["id"]
                        proposals.append(proposal)
                        proposal_calls.append({
                            "proposal_id": proposal["proposal_id"],
                            "tool_call_id": item["id"],
                        })
                except ValueError as exc:
                    error_type = "invalid_arguments"
                    result = {"status": "error", "error_type": error_type, "message": str(exc)[:300]}
                except Exception as exc:
                    error_type = getattr(exc, "category", None) or type(exc).__name__
                    message_text = str(exc)[:300] if getattr(exc, "category", None) else "工具执行失败，请检查服务配置后重试。"
                    result = {"status": "error", "error_type": error_type, "message": message_text}

                elapsed = round((time.perf_counter() - tool_started) * 1000, 2)
                stage_ms[f"tool_{len(tool_trace) + 1}"] = elapsed
                if proposal is not None:
                    tool_trace.append({
                        "name": name, "status": "awaiting_confirmation",
                        "elapsed_ms": elapsed, "result_count": 0,
                        "proposal_id": proposal["proposal_id"],
                        "side_effecting": proposal["side_effecting"],
                    })
                    continue

                status = (result or {}).get("status", "succeeded")
                result_count = (result or {}).get("result_count", len((result or {}).get("results", [])))
                sources = (result or {}).get("results", [])
                source_urls = [
                    source["url"] for source in sources
                    if isinstance(source, dict) and isinstance(source.get("url"), str)
                ]
                if error_type:
                    status = "failed"
                tool_trace.append({
                    "name": name, "status": status, "elapsed_ms": elapsed,
                    "result_count": result_count, "error_type": error_type,
                    "source_urls": source_urls,
                })
                round_results.append({
                    "role": "tool", "tool_call_id": item["id"],
                    "content": self._tool_result_content(result or {}),
                })

            messages.extend(round_results)
            if proposals:
                return (
                    message.content or "需要确认 MCP 工具调用后才能继续。",
                    "awaiting_tool_approval", proposals, tool_call_data, tool_trace,
                    rounds, tool_warning, self._safe_tool_context(messages), proposal_calls,
                )
            if rounds >= max_rounds:
                force_final = True

    def run_turn(self, task_id: str, text: str = "", image_paths=None, audio_path: str | None = None,
                 display_text: str | None = None, skill_id: str | None = None,
                 on_tool_status: Callable[[str], None] | None = None) -> AgentTurn:
        run_id, started = str(uuid.uuid4()), time.perf_counter()
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        policy = self.policy_service.get_active() if self.policy_service is not None else None
        if task.status == "paused":
            task = self.tasks.resume(task_id)
        elif task.status != "active":
            raise ValueError(f"task is {task.status} and cannot accept a new turn")
        metadata, content_parts = [], []
        stage_ms = {}
        memories = []
        evidence = []
        rag_trace = None
        tool_calls_trace = []
        selected_skill = None
        user_text = (text or "").strip()
        stored_user_text = user_text
        try:
            selected_skill = self.skills.get_skill(skill_id) if skill_id else None
            if skill_id and selected_skill is None:
                raise ValueError(f"unknown skill: {skill_id}")
            if len(user_text) > 20000:
                raise ValueError("单次文字输入不能超过 20,000 字符。")
            if audio_path:
                stage_started = time.perf_counter()
                try:
                    transcript, audio_meta = self.media.process_audio(audio_path)
                finally:
                    stage_ms["asr"] = round((time.perf_counter() - stage_started) * 1000, 2)
                user_text = "\n".join(part for part in (user_text, f"[Audio transcript]\n{transcript}") if part)
                metadata.append(audio_meta)
                if len(user_text) > 20000:
                    raise ValueError("文字与音频转写合计不能超过 20,000 字符。")
            if image_paths:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                if len(image_paths) > 4:
                    raise ValueError("单轮最多上传 4 张图片。")
                for image_path in image_paths:
                    block, image_meta = self.media.process_image(image_path)
                    content_parts.append(block)
                    metadata.append(image_meta)
            if not user_text and not content_parts:
                raise ValueError("请输入文字，或附上一张图片/一段音频。")
            stored_user_text = (display_text or user_text).strip()
            if user_text:
                content_parts.insert(0, {"type": "text", "text": user_text})
            query = user_text or "请描述附件内容，并结合当前任务上下文给出下一步。"
            stage_started = time.perf_counter()
            if policy is None:
                memories = self.memory.retrieve(query, task_id=task_id, limit=8)
            else:
                memories = self.memory.retrieve(query, task_id=task_id, limit=policy.memory_top_k,
                                               layer_weights=dict(policy.layer_weights),
                                               time_decay_days=policy.time_decay_days,
                                               allowed_tags=policy.allowed_memory_tags,
                                               excluded_tags=policy.excluded_memory_tags)
            stage_ms["memory_retrieval"] = round((time.perf_counter() - stage_started) * 1000, 2)
            context = self.tasks.context(task_id)
            evidence = []
            rag_trace = None
            if self.rag_service is not None and self._looks_like_knowledge_question(query):
                rag_started = time.perf_counter()
                try:
                    rag_run = self.rag_service.ask(query)
                    evidence = rag_run.evidence[:5] if rag_run.status == "succeeded" else []
                    rag_trace = {"run_id": rag_run.run_id, "status": rag_run.status,
                                 "citations": rag_run.citations, "corpus_version": rag_run.corpus_version}
                except Exception:
                    evidence = []
                finally:
                    stage_ms["rag"] = round((time.perf_counter() - rag_started) * 1000, 2)
            system = self._system_prompt(context, memories, evidence)
            if selected_skill is not None:
                system += (f"\n\n当前工作流：{selected_skill.title} v{selected_skill.version}\n"
                           f"工作流要求：{selected_skill.instructions}\n"
                           f"工作流工具允许列表：{', '.join(selected_skill.allowed_tools)}")
            if policy is not None:
                if policy.prompt_suffix:
                    system += "\n\nAgent policy 指令：\n" + policy.prompt_suffix
                if policy.prompt_template_ref == "default-v1":
                    system += "\n当前提示模板：default-v1"
                if policy.output_constraints.get("require_citations"):
                    system += "\n策略要求：对知识库结论附带对应证据标识。"
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content_parts}]
            (answer, status, proposals, tool_call_data, tool_calls_trace, tool_rounds,
             tool_warning, tool_context, proposal_calls) = self._run_tool_loop(
                messages, task_id, run_id, stage_ms, user_text, on_tool_status, selected_skill, policy,
                trace_collector=tool_calls_trace,
            )
            if policy is not None:
                max_chars = policy.output_constraints["max_output_chars"]
                answer = answer[:max_chars]
            stage_ms["deepseek"] = round(sum(
                duration for key, duration in stage_ms.items() if key.startswith("deepseek_round_")
            ), 2)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            trace = {"run_id": run_id, "task_id": task_id, "status": status, "elapsed_ms": elapsed,
                     "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), "memory_hits": [{"memory_id": hit.item.memory_id,
                     "layer": hit.item.layer, "content": hit.item.content, "score": hit.score,
                     "explanation": hit.explanation} for hit in memories],
                     "rag_trace": rag_trace,
                     "rag_evidence": [{"chunk_id": item.get("chunk_id"), "filename": item.get("filename"),
                                       "page": item.get("page"), "score": item.get("score"),
                                       "content": item.get("content", "")[:700]}
                                      for item in evidence], "attachments": metadata,
                     "tool_proposals": proposals, "tool_calls": tool_calls_trace,
                     "tool_rounds": tool_rounds, "skill_id": skill_id,
                     "skill_version": selected_skill.version if selected_skill else None, "stage_ms": stage_ms}
            if policy is not None:
                policy_payload = {
                    "policy_id": policy.policy_id, "version": policy.version,
                    "prompt_suffix": policy.prompt_suffix, "prompt_template_ref": policy.prompt_template_ref,
                    "memory_top_k": policy.memory_top_k, "layer_weights": dict(policy.layer_weights),
                    "time_decay_days": policy.time_decay_days, "allowed_memory_tags": list(policy.allowed_memory_tags),
                    "excluded_memory_tags": list(policy.excluded_memory_tags), "tool_allowlist": list(policy.tool_allowlist),
                    "max_tool_rounds": policy.max_tool_rounds,
                    "output_constraints": dict(policy.output_constraints), "checker_ids": list(policy.checker_ids),
                }
                trace["policy_id"] = policy.policy_id
                trace["policy_version"] = policy.version
                trace["policy_hash"] = hashlib.sha256(json.dumps(policy_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            if tool_warning:
                trace["mcp_warning"] = tool_warning
            self.store.append_event(task_id, "agent_turn", {"user_text": stored_user_text[:3000], "answer": answer[:6000],
                                    "trace": trace, "tool_call": tool_call_data,
                                    "tool_context": tool_context,
                                    "tool_proposal_calls": proposal_calls}, run_id)
            checkpoint = {"last_run_id": run_id, "last_user_message": stored_user_text[:3000],
                          "last_assistant_message": answer[:6000], "last_progress": answer[:1200],
                          "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self.tasks.checkpoint(task_id, checkpoint)
            # Conversation episodes are retained as reviewable candidates, not silently promoted.
            episode_content = f"Task: {task.title}\nUser: {stored_user_text[:2000]}\nAssistant: {answer[:3000]}"
            episode = MemoryItem(str(uuid.uuid4()), "episodic", episode_content,
                                 "pending", task_id, run_id, 0.5, {"kind": "turn_summary", "attachments": metadata},
                                 self.memory.make_embedding(episode_content))
            self.store.put_memory(episode)
            trace["memory_candidate_id"] = episode.memory_id
            # Persist the candidate reference together with the run trace so it can be audited after restart.
            self.store.append_event(task_id, "agent_turn_memory_candidate",
                                    {"run_id": run_id, "memory_candidate_id": episode.memory_id}, run_id)
            return AgentTurn(run_id, task_id, status, answer, trace, proposals)
        except Exception as exc:
            error_type = type(exc).__name__
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            user_message = (str(exc) if isinstance(exc, ValueError) or
                            (isinstance(exc, RuntimeError) and "faster-whisper" in str(exc))
                            else f"本轮未完成：{error_type}。请展开运行详情检查阶段耗时与错误类型。")
            failed_trace = {"run_id": run_id, "task_id": task_id, "status": "failed",
                            "error_type": error_type, "elapsed_ms": elapsed_ms, "stage_ms": stage_ms,
                            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                            "memory_hits": [{"memory_id": hit.item.memory_id, "layer": hit.item.layer,
                                             "content": hit.item.content, "score": hit.score,
                                             "explanation": hit.explanation} for hit in memories],
                            "rag_trace": rag_trace,
                            "rag_evidence": [{"chunk_id": item.get("chunk_id"), "filename": item.get("filename"),
                                              "page": item.get("page"), "score": item.get("score"),
                                              "content": item.get("content", "")[:700]} for item in evidence],
                            "attachments": metadata, "tool_calls": tool_calls_trace,
                            "skill_id": skill_id,
                            "skill_version": selected_skill.version if selected_skill else None,
                            "user_message": user_message}
            if policy is not None:
                policy_payload = {
                    "policy_id": policy.policy_id, "version": policy.version,
                    "prompt_suffix": policy.prompt_suffix, "prompt_template_ref": policy.prompt_template_ref,
                    "memory_top_k": policy.memory_top_k, "layer_weights": dict(policy.layer_weights),
                    "time_decay_days": policy.time_decay_days, "allowed_memory_tags": list(policy.allowed_memory_tags),
                    "excluded_memory_tags": list(policy.excluded_memory_tags), "tool_allowlist": list(policy.tool_allowlist),
                    "max_tool_rounds": policy.max_tool_rounds,
                    "output_constraints": dict(policy.output_constraints), "checker_ids": list(policy.checker_ids),
                }
                failed_trace["policy_id"] = policy.policy_id
                failed_trace["policy_version"] = policy.version
                failed_trace["policy_hash"] = hashlib.sha256(json.dumps(
                    policy_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            failure = {"error_type": error_type, "elapsed_ms": elapsed_ms, "stage_ms": stage_ms,
                       "user_text": stored_user_text[:3000], "answer": user_message[:6000],
                       "trace": failed_trace}
            self.store.append_event(task_id, "agent_turn_failed", failure, run_id)
            return AgentTurn(run_id, task_id, "failed", user_message, failed_trace)

    def execute_tool_proposal(self, proposal_id: str, confirmed: bool) -> dict:
        outcome = self.tools.execute_call(proposal_id, user_confirmed=confirmed)
        if outcome.get("status") != "executed":
            return outcome
        proposal = self.store.get_tool_proposal(proposal_id)
        source = next((event for event in self.store.list_events(proposal["task_id"], limit=300)
                       if event.run_id == proposal["run_id"] and event.event_type == "agent_turn"), None)
        if source is None:
            outcome["continuation_error"] = "missing_tool_call_context"
            return outcome

        payload = source.payload
        proposal_calls = payload.get("tool_proposal_calls") or []
        if not proposal_calls:
            outcome["continuation_error"] = "missing_tool_call_context"
            return outcome
        related = []
        for item in proposal_calls:
            stored = self.store.get_tool_proposal(item.get("proposal_id", ""))
            if stored is None:
                outcome["continuation_error"] = "missing_tool_proposal"
                return outcome
            related.append((item, stored))
        if any(stored["status"] == "pending" for _, stored in related):
            outcome["status"] = "awaiting_tool_approval"
            return outcome

        task_id = proposal["task_id"]
        source_trace = payload.get("trace", {})
        source_policy = None
        if self.policy_service is not None and hasattr(self.policy_service, "get"):
            source_policy = self.policy_service.get(source_trace.get("policy_id", ""))
        memories = self.memory.retrieve(
            payload.get("user_text", ""), task_id=task_id,
            limit=source_policy.memory_top_k if source_policy else 8,
            layer_weights=dict(source_policy.layer_weights) if source_policy else None,
            time_decay_days=source_policy.time_decay_days if source_policy else None,
            allowed_tags=source_policy.allowed_memory_tags if source_policy else (),
            excluded_tags=source_policy.excluded_memory_tags if source_policy else (),
        )
        system = self._system_prompt(self.tasks.context(task_id), memories, [])
        if source_policy and source_policy.prompt_suffix:
            system += "\n\n策略补充：\n" + source_policy.prompt_suffix
        context = payload.get("tool_context") or []
        messages = [{"role": "system", "content": system}, *context]
        for item, stored in related:
            if stored["status"] == "executed":
                result = stored.get("result") or {}
            else:
                result = {"status": "rejected", "message": "用户拒绝了该工具调用。"}
            messages.append({
                "role": "tool",
                "tool_call_id": item["tool_call_id"],
                "content": json.dumps(result, ensure_ascii=False)[:16000],
            })
        try:
            response = self._client().chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                messages=messages,
                stream=False,
            )
            answer = response.choices[0].message.content or "工具已执行，但模型没有生成后续说明。"
            if source_policy:
                max_chars = source_policy.output_constraints["max_output_chars"]
                answer = answer[:max_chars]
            self.store.append_event(task_id, "agent_tool_followup",
                                    {"run_id": proposal["run_id"], "tool_proposal_id": proposal_id,
                                     "answer": answer[:6000]}, proposal["run_id"])
            self.tasks.checkpoint(task_id, {"last_assistant_message": answer[:6000],
                                 "last_progress": answer[:1200], "last_tool_proposal_id": proposal_id})
            outcome["final_answer"] = answer
        except Exception as exc:
            outcome["continuation_error"] = type(exc).__name__
        return outcome

    def reject_tool_proposal(self, proposal_id: str) -> dict:
        proposal = self.store.get_tool_proposal(proposal_id)
        if proposal is None or proposal["status"] != "pending":
            raise ValueError("tool proposal is missing or no longer pending")
        self.store.finish_tool_proposal(proposal_id, "rejected", {"reason": "user_rejected"})
        self.store.append_event(proposal["task_id"], "mcp_tool_rejected",
                                {"proposal_id": proposal_id, "tool_name": proposal["tool_name"]}, proposal["run_id"])
        return {"status": "rejected", "proposal_id": proposal_id}

    def correct_turn(self, task_id: str, run_id: str, correction: str) -> AgentTurn:
        correction = (correction or "").strip()
        if not correction:
            raise ValueError("请填写需要纠正或补充的内容")
        prior = next((event for event in self.store.list_events(task_id, limit=200)
                      if event.run_id == run_id and event.event_type == "agent_turn"), None)
        if prior is None:
            raise ValueError("找不到对应的 Agent 运行记录")
        self.store.append_event(task_id, "user_correction", {"corrects_run_id": run_id,
                                "correction": correction[:3000]})
        candidate = self.memory.propose("semantic", correction, source_task_id=task_id,
                                        source_run_id=run_id, confidence=0.8,
                                        metadata={"kind": "user_correction", "review_required": True})
        previous = prior.payload
        prompt = ("请根据用户刚才的纠正重新回答。把纠正视为本轮可信输入，但不要将其扩展成未经证实的事实。\n"
                  f"上一轮问题：{previous.get('user_text', '')}\n上一轮回答：{previous.get('answer', '')}\n"
                  f"用户纠正：{correction}")
        turn = self.run_turn(task_id, prompt, display_text=f"用户纠错：{correction}")
        trace = {**turn.trace, "correction_candidate_id": candidate.memory_id, "corrects_run_id": run_id}
        return AgentTurn(turn.run_id, turn.task_id, turn.status, turn.answer, trace, turn.pending_tool_calls)

    @staticmethod
    def _looks_like_knowledge_question(query: str) -> bool:
        hints = ("文档", "知识库", "资料", "根据", "文件", "报告", "PDF")
        return any(word in query for word in hints)

    @staticmethod
    def _system_prompt(task: dict, memories, evidence) -> str:
        memory_text = "\n".join(f"[{hit.item.layer} | {hit.item.memory_id}] {hit.item.content}" for hit in memories)
        evidence_text = "\n".join(f"[RAG {item.get('chunk_id', '')}] {item.get('content', '')[:1200]}" for item in evidence)
        return ("你是 EdgeMind，一个面向个人长期任务的本地优先 Agent。严格区分已确认事实与推断；"
                "若上下文不足应提出澄清。不要声称已执行工具，除非工具结果已返回。回答末尾简要给出下一步。\n\n"
                f"当前任务：{json.dumps(task, ensure_ascii=False)}\n\n"
                f"已批准的相关记忆：\n{memory_text or '无'}\n\n"
                f"个人知识库证据：\n{evidence_text or '本轮未检索到文档证据'}")
