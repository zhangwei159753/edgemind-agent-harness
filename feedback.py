import json, os, re
from typing import Dict, Any, List
from openai import OpenAI

class FeedbackLoop:
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434/v1"
        self.ollama_model = "deepseek-r1:7b"
        self.client = OpenAI(api_key="ollama", base_url=self.ollama_base_url)

    # ========= 新增：日志持久化 =========
    def log_feedback(self,
                     question: str,
                     draft: str,
                     loop_id: str,
                     chunks: List[Dict[str, Any]]) -> None:
        """把每一轮修正结果追加写入 feedback_log.jsonl"""
        log_file = "feedback_log.jsonl"
        record = {
            "question": question,
            "draft": draft,
            "loop": loop_id,
            "chunks": chunks,
            "timestamp": str(os.times())
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ============= 新增：推理链结构 =============
    def _pack_reasoning_trace(self,
                              question: str,
                              chunks: List[Dict[str, Any]],              # 键为字符串，值为任意类型
                              reasoning: str,
                              draft: str) -> Dict[str, Any]:
        """把检索、推理、答案封装成可追溯的结构"""
        return {
            "question": question,
            "chunks": chunks,          # 带分数、来源
            "reasoning": reasoning,
            "draft": draft,
            "timestamp": str(os.times())
        }

    # ============= 1. 自我批判（Self-Critique） =============
    def _self_critique(self,
                       trace: Dict[str, Any],
                       feedback: str) -> Dict[str, Any]:
        """让模型逐行检查自己的推理链"""
        prompt = (
            "你是一名严谨的金融分析师，请根据用户反馈，逐行检查下列推理链，指出每一步是否有误，并给出修正建议，以及是否需要重检索。\n"
            f"用户反馈：{feedback}\n推理链：{trace['reasoning']}\n原答案：{trace['draft']}"
            "\n请严格按如下的JSON格式输出：\n"
            '{"critiques":[{"issue":"这里写该推理过程中存在的问题","fix":"这里写你的修正建议"}],"needs_retrieval":true或false,"new_keywords":["涉及用户反馈的关键词"]}\n'
            # '{"critiques":["issue":"【这里写该推理过程中存在的问题】","fix":"【这里写你的修正建议】"}],"needs_retrieval":true/false,"new_keywords":["涉及用户反馈的关键词"]}\n'
            f"请必须按照上面的JSON格式输出，确保输出内容为合法JSON字符串，不要输出多余内容或者别的格式的JSON字符串。\n"
        )
        raw = self._chat(prompt)
        # # 去掉 ```json ... ``` 外壳
        # raw = re.sub(r'^\s*```(?:json)?\s*\n?', '', raw, flags=re.IGNORECASE)
        # raw = re.sub(r'\n?```\s*$', '', raw)
        # raw = re.sub(r'\s+', ' ', raw)  # 合并多余空白
        # raw = raw.replace('\u3000', ' ')  # 替换全角空格
        # # print("自我批判模型原始回复1：", raw)

        # === 新增：多重清洗 + 容错 ===
        # 1. 去掉 ```json / ``` 外壳
        raw = re.sub(r'^\s*```(?:json)?\s*\n?', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\n?```\s*$', '', raw)
        # 2. 去掉 <think>...</think>
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        # 3. 合并多余空白
        raw = re.sub(r'\s+', ' ', raw).strip()

        # 4. 只保留第一个合法 JSON 对象
        match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
        if not match:
            print("【调试】_self_critique 未找到 JSON 对象")
            return {"critiques": [], "needs_retrieval": False, "new_keywords": []}

        raw = match.group(0)

        try:
            parsed = json.loads(raw)
            print("\n【自我批判】 解析结果：", parsed)
            return parsed
        except Exception as e:
            print("\n【调试】_self_critique 解析异常：", e)
            return {"critiques": [], "needs_retrieval": False, "new_keywords": []}

    # ============= 2. 定向重检索 =============
    def _directed_retrieve(self,
                           keywords: List[str],
                           original_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """仅针对关键词做补充检索（这里简化：用 BM25 再查 3 条）"""
        # 真实场景可复用 rag.search_hybrid
        from rank_bm25 import BM25Okapi
        # 中文分词 + 去停用词
        import jieba
        # print("\n【调试1】")
        stop_words = {'的', '了', '和', '是', '在', '为', '与', '等'}  # 可扩展
        def tokenize(text):
            return [w for w in jieba.cut_for_search(text) if w.strip() and w not in stop_words]
        corpus = [tokenize(c['content']) for c in original_chunks if c.get('content')]

        # corpus = []
        # for idx, c in enumerate(original_chunks):
        #     content = c.get('content')
        #     print(f"[DEBUG] chunk[{idx}] content={repr(content)}")
        #     if not content:
        #         continue
        #     try:
        #         tokens = tokenize(content)
        #         corpus.append(tokens)
        #     except Exception as e:
        #         print(f"[ERROR] chunk[{idx}] tokenize fail: {e}")
        #         continue

        bm25 = BM25Okapi(corpus)
        new_docs = []
        for kw in keywords:
            scores = bm25.get_scores(kw.split())
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]  # 取前3得分最高的
            new_docs.extend([original_chunks[i] for i in top_idx if scores[i] > 0])
        # 去重
        seen = {(c['metadata']['file_name'], c['metadata']['page']) for c in new_docs}
        return [c for c in new_docs if (c['metadata']['file_name'], c['metadata']['page']) in seen]

    # ============= 3. 定向优化 =============
    def _refine_answer(self,
                       trace: Dict[str, Any],
                       critiques: List[Dict[str, Any]],
                       new_chunks: List[Dict[str, Any]],
                       feedback: str) -> Dict[str, Any]:
        """根据批判意见 + 新片段 重新生成"""
        context = "\n".join([
            f"[文件名]{c['metadata']['file_name']} [页码]{c['metadata']['page']}\n{c['content']}"
            # for c in (trace['chunks'] + new_chunks)     # 有点异议，这里应该是直接为new_chunks？
            for c in new_chunks  # 有点异议，这里应该是直接为new_chunks？
        ])
        prompt = (
            "你是一名严谨的金融分析师，请根据用户反馈、批判意见、新检索到的内容、原问题，去修正原答案得到新答案。\n"
            f"用户反馈：{feedback}\n批判意见：{json.dumps(critiques, ensure_ascii=False)}\n新检索到的内容：{context}\n原问题：{trace['question']}\n原答案：{trace['draft']}\n"
            "请严格按 JSON 格式输出：\n"
            '{"reasoning":"你的逐步修正过程","answer":"你的新答案","filename":"文件名.pdf","page":"页码"}\n'
            f"请必须按照上面的JSON格式输出，确保输出内容为合法JSON字符串，不要输出多余内容或者别的格式的JSON字符串。\n"
        )
        raw = self._chat(prompt)
        raw = re.sub(r'^\s*```(?:json)?\s*\n?', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\n?```\s*$', '', raw)
        raw = re.sub(r'\s+', ' ', raw)  # 合并多余空白
        raw = raw.replace('\u3000', ' ')  # 替换全角空格
        # print("自我批判模型原始回复2：", raw)

        # ---- 新增兜底：如果内容不是合法 JSON，直接返回空结构 ----
        if not re.match(r'^\s*\{.*}\s*$', raw, flags=re.DOTALL):
            return {"critiques": [], "needs_retrieval": False, "new_keywords": []}

        try:
            parsed = json.loads(raw.strip())
            # print("\n【调试】_self_critique 解析结果：", parsed)
            return parsed
        except Exception as e:
            print("\n【调试】_self_critique 解析异常：", e)
            return {"critiques": [], "needs_retrieval": False, "new_keywords": []}

        # try:
        #     return json.loads(raw)
        # except:
        #     return {"reasoning": "", "answer": raw, "filename": "", "page": ""}

    # ============= 4. 多轮迭代闭环 =============
    def regenerate_with_loop(self,
                               question: str,
                               trace: Dict[str, Any],
                               feedback: str,
                               max_loop: int = 3) -> Dict[str, Any]:
        """
        对外统一入口：自我批判 → 定向优化 → 多轮迭代
        返回结构同原来，但内部可能已迭代多次
        """
        for loop in range(max_loop):
            print(f"\n进入 regenerate_with_loop 第{loop+1}轮")
            critique = self._self_critique(trace, feedback)
            if not critique['critiques']:
                break  # 模型自认无误

            new_chunks = []
            # print("\n【修正后回答1】")
            if critique.get('needs_retrieval') and critique.get('new_keywords'):
                new_chunks = self._directed_retrieve(critique['new_keywords'], trace['chunks'])
                print(f"\n定向重检索到{len(new_chunks)}个新chunks")
                # print(trace['chunks'])

            # print("\n【修正后回答2】")
            refined = self._refine_answer(trace, critique['critiques'], new_chunks, feedback)
            print(f"\n【定向优化】:{refined}")    # 即修正后的回答，_refine_answer函数的输出
            trace.update({
                "reasoning": refined['reasoning'],
                "draft": refined['answer'],
                "chunks": trace['chunks'] + new_chunks
            })

            # 记录每一轮
            self.log_feedback(question, trace['draft'], f"loop-{loop+1}", trace['chunks'])

        return {
            "answer": trace['draft'],
            "reasoning": trace['reasoning'],
            "filename": refined.get('filename', ''),
            "page": refined.get('page', '')
        }

    # ============= 内部调用 LLM =============
    def _chat(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024
        )
        raw = resp.choices[0].message.content.strip()
        return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
