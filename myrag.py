import json
import os
import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import concurrent.futures
import random
import numpy as np
from numpy import dot
import spacy
from numpy.linalg import norm
import jieba
from FlagEmbedding import FlagReranker
from get_text_embedding import get_text_embedding
from dotenv import load_dotenv
from openai import OpenAI
from edgemind.providers import DeepSeekSettings, create_deepseek_client
from rank_bm25 import BM25Okapi
from feedback import FeedbackLoop
from intent_inheritance import detect_intent_inheritance, _extract_keywords
# 统一加载项目根目录的.env
load_dotenv()

try:
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("警告：未安装langchain，将使用简单的内存存储方案")
    class ConversationBufferMemory:
        def __init__(self, **kwargs):
            self.messages = []
        def add_user_message(self, message):
            self.messages.append({"role": "user", "content": message})
        def add_ai_message(self, message):
            self.messages.append({"role": "assistant", "content": message})
        @property
        def buffer(self):
            return self.messages

class PageChunkLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
    def load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

# ----- 辅助函数（类外） -----
def contains_pronoun_or_ellipsis(question: str) -> bool:
    """检测是否存在代词/省略提示的词（中文为主）"""
    pronouns = ["它", "他", "她", "这", "那个", "那些", "这些", "那", "这样", "哪一个", "呢", "吗"]
    time_words = ["上周", "上个月", "本月", "这月", "这周", "上季度", "去年", "今年", "上半年", "下半年"]
    for w in pronouns + time_words:
        if w in question:
            print(f"【省略内容解析】发现问题中有存在的指代/省略词！直接进行问题改写！")
            return True
    # 简单的以疑问词结尾（“呢”）也可判定
    if question.endswith("呢") or question.endswith("吗"):
        print(f"【省略内容解析】发现问题中有存在的指代/省略词！直接进行问题改写！")
        return True
    # === 新增：意图缺省检查 ===
    keywords, missing, missing_parts = _extract_keywords(question)
    if missing:
        print(f"【意图缺省】检测到缺失成分：{missing_parts}，将尝试改写问题")
        return True
    return False

class EmbeddingModel:
    def __init__(self, batch_size: int = 64):
        self.api_key = os.getenv('LOCAL_API_KEY')
        self.base_url = os.getenv('LOCAL_BASE_URL')
        self.embedding_model = os.getenv('LOCAL_EMBEDDING_MODEL')
        self.batch_size = batch_size
        if not self.api_key or not self.base_url:
            raise ValueError('请在.env中配置LOCAL_API_KEY和LOCAL_BASE_URL')

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return get_text_embedding(
            texts,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size
        )

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.chunks = []
        # --- 新增 ---
        self.bm25 = None  # BM25 模型
        self.tokenized_chunks = []  # 预先分好词的文本

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        # --- 新增：构建 BM25 ---
        # 使用 jieba 精确分词
        self.tokenized_chunks = [
            list(jieba.cut_for_search(c['content']))  # 搜索引擎模式，速度快
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        from numpy import dot
        from numpy.linalg import norm
        import numpy as np
        if not self.embeddings:
            return []
        emb_matrix = np.array(self.embeddings)
        query_emb = np.array(query_embedding)
        sims = emb_matrix @ query_emb / (norm(emb_matrix, axis=1) * norm(query_emb) + 1e-8)
        idxs = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        # 同样用 jieba 分词
        tokens = list(jieba.cut_for_search(query))
        scores = self.bm25.get_scores(tokens)
        idxs = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]


class SimpleRAG:
    def __init__(self, chunk_json_path: str, model_path: str = None, batch_size: int = 32):
        self.loader = PageChunkLoader(chunk_json_path)
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        self.vector_store = SimpleVectorStore()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.feedback = FeedbackLoop()    # 新增人类反馈思维链
        self.latest_trace = None  # 新增：保存最近一次完整推理链
        self.last_retrieved_chunks = None
        self.auto_save_history = False  # 默认关闭
        # 初始化 FlagReranker（加载一次就行）
        local_model_path = "./bge-reranker-v2-m3"  # 替换为你的实际路径
        self.reranker = FlagReranker(
            local_model_path,
            use_fp16=False,
            devices=["cuda:0"]
        )

    def search_hybrid(self, question: str, top_k_vec: int = 10, top_k_bm25: int = 10) -> List[Dict[str, Any]]:
        """
        混合检索：向量 + BM25，各取 top_k，合并去重后返回
        """
        # 向量检索
        q_emb = self.embedding_model.embed_text(question)
        vec_results = self.vector_store.search(q_emb, top_k=top_k_vec)

        # BM25 检索
        bm25_results = self.vector_store.search_bm25(question, top_k=top_k_bm25)

        # 合并去重（保持顺序）
        seen = set()
        merged = []
        for chunk in vec_results + bm25_results:
            cid = (chunk['metadata']['file_name'], chunk['metadata']['page'], chunk['content'])
            if cid not in seen:
                seen.add(cid)
                merged.append(chunk)
        return merged

    # 新增：实体级联检索
    def search_entity_cascade(self, question: str,
                              top_k_per_ent: int = 10,
                              top_k_final: int = 10) -> List[Dict[str, Any]]:
        """
        1) spaCy 抽取实体
        2) 依次对每个实体做检索/过滤
        3) 最后再对整个问题做一轮检索
        返回合并后的 chunk 列表（已去重）
        """
        try:
            nlp = spacy.load("zh_core_web_lg")
        except OSError:
            raise RuntimeError("请先安装 spaCy 中文模型：python -m spacy download zh_core_web_lg")

        doc = nlp(question)
        entities = [ent.text for ent in doc.ents]
        print(f"【实体识别】{entities}")

        # 第一轮：按实体依次检索
        current_chunks = None
        for idx, ent in enumerate(entities, 1):
            print(f"第{idx}次检索：{ent}")
            hits = self.search_hybrid(ent, top_k_vec=top_k_per_ent, top_k_bm25=top_k_per_ent)

            if current_chunks is None:
                # 第一次：用 dict 去重
                current_chunks = {(c['metadata']['file_name'], c['metadata']['page']): c
                                  for c in hits}
            else:
                # 后续：只保留同时包含当前实体的 chunk
                ent_hits = {(c['metadata']['file_name'], c['metadata']['page']): c
                            for c in hits}
                # 取交集
                current_chunks = {k: v for k, v in current_chunks.items() if k in ent_hits}

            print(f"→ 当前保留 {len(current_chunks)} 个 chunk")    # search_hybrid中各自检索回5个，最后得到5+5=10个

        # 第二轮：对整个问题再做一次检索
        print("最终问题检索：", question)
        final_hits = self.search_hybrid(question,
                                        top_k_vec=top_k_final,
                                        top_k_bm25=top_k_final)

        # 合并去重（优先保留实体层结果）
        merged = dict(current_chunks or {})
        for c in final_hits:
            key = (c['metadata']['file_name'], c['metadata']['page'])
            merged.setdefault(key, c)

        return list(merged.values())

    # 源代码：每次都重新构建向量数据库
    # def setup(self):
    #     print("加载所有页chunk...")
    #     chunks = self.loader.load_chunks()
    #     print(f"共加载 {len(chunks)} 个chunk")
    #     print("生成嵌入...")
    #     embeddings = self.embedding_model.embed_texts([c['content'] for c in chunks])
    #     print("存储向量...")
    #     self.vector_store.add_chunks(chunks, embeddings)
    #     print("RAG向量库构建完成！")

    def setup(self):
        import pickle
        cache_file = "vector_store.pkl"

        if os.path.exists(cache_file):
            print("加载已存在的向量数据库...")
            with open(cache_file, 'rb') as f:
                self.vector_store = pickle.load(f)

            # 动态补建缺失字段
            if not hasattr(self.vector_store, 'bm25') or self.vector_store.bm25 is None:
                print("补建 BM25 索引...")
                self.vector_store.tokenized_chunks = [
                    list(jieba.cut_for_search(c['content']))
                    for c in self.vector_store.chunks
                ]
                self.vector_store.bm25 = BM25Okapi(self.vector_store.tokenized_chunks)
        else:
            print("首次构建向量数据库...")
            chunks = self.loader.load_chunks()
            embeddings = self.embedding_model.embed_texts([c['content'] for c in chunks])
            self.vector_store.add_chunks(chunks, embeddings)

            # 保存到缓存
            with open(cache_file, 'wb') as f:
                pickle.dump(self.vector_store, f)

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        q_emb = self.embedding_model.embed_text(question)
        results = self.vector_store.search(q_emb, top_k)
        return {
            "question": question,
            "chunks": results
        }

    def save_conversation_history(self, file_path: str = "conversation_history.json"):
        """保存对话历史到文件"""
        try:
            history_data = []
            if hasattr(self.memory, 'buffer') and self.memory.buffer:
                for msg in self.memory.buffer:
                    if LANGCHAIN_AVAILABLE and hasattr(msg, 'type'):
                        history_data.append({
                            "role": msg.type,
                            "content": msg.content
                        })
                    else:
                        history_data.append({
                            "role": msg.get('role', 'user'),
                            "content": msg.get('content', '')
                        })

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            print(f"对话历史已保存到 {file_path}")
        except Exception as e:
            print(f"保存对话历史失败: {str(e)}")

    def load_conversation_history(self, file_path: str = "conversation_history.json"):
        """从文件加载对话历史"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)

                # 清空当前记忆
                self.memory = ConversationBufferMemory(return_messages=True)

                # 重新添加历史消息
                for msg in history_data:
                    if LANGCHAIN_AVAILABLE:
                        from langchain.schema import HumanMessage, AIMessage
                        if msg.get('role') == 'user':
                            self.memory.chat_memory.add_user_message(msg.get('content', ''))
                        elif msg.get('role') == 'assistant':
                            self.memory.chat_memory.add_ai_message(msg.get('content', ''))
                    else:
                        if msg.get('role') == 'user':
                            self.memory.add_user_message(msg.get('content', ''))
                        elif msg.get('role') == 'assistant':
                            self.memory.add_ai_message(msg.get('content', ''))

                print(f"已从 {file_path} 加载对话历史")
                return True
            else:
                print(f"未找到对话历史文件 {file_path}")
                return False
        except Exception as e:
            print(f"加载对话历史失败: {str(e)}")
            return False

    def maintain_sliding_window_memory(self, max_window_size: int = 3):
        """
        滑动窗口记忆管理：维护固定大小的对话窗口
        当对话数量超过max_window_size时，从队首移除最旧的一条对话
        """
        try:
            # 获取当前消息数量
            if LANGCHAIN_AVAILABLE and hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                messages = self.memory.chat_memory.messages
                if len(messages) > max_window_size * 2:  # *2是因为每条对话包含用户和助手各一条消息
                    # 移除最旧的对话（用户消息和助手回复）
                    messages.pop(0)  # 移除最旧的用户消息
                    if messages:
                        messages.pop(0)  # 移除对应的助手回复
            elif hasattr(self.memory, 'buffer'):
                # 兼容简单内存格式
                messages = self.memory.buffer
                if len(messages) > max_window_size * 2:
                    messages.pop(0)  # 移除最旧的消息
                    if messages:
                        messages.pop(0)  # 移除对应的回复
        except Exception as e:
            print(f"维护滑动窗口记忆时出错: {str(e)}")

    def add_conversation_pair(self, user_message: str, assistant_message: str):
        """
        添加对话对到滑动窗口记忆
        先添加新对话，然后维护窗口大小
        """
        # 添加用户消息
        if LANGCHAIN_AVAILABLE:
            self.memory.chat_memory.add_user_message(user_message)
        else:
            self.memory.add_user_message(user_message)

        # 添加助手回复
        if LANGCHAIN_AVAILABLE:
            self.memory.chat_memory.add_ai_message(assistant_message)
        else:
            self.memory.add_ai_message(assistant_message)

        # 维护滑动窗口大小
        self.maintain_sliding_window_memory()

    def rewrite_question_with_context(self, question: str, max_prompt_tokens: int = 512) -> str:
        """
        若问题中含有指代/省略，尝试将其改写为独立的完整问题。
        策略：
         - 若检测到代词或相对时间，优先调用本地模型做补全（结合最近的若干条消息）
         - 否则返回原问题
        """
        if not contains_pronoun_or_ellipsis(question):
            print(f"【省略内容解析】没有发现指代/省略词！直接返回原问题！")
            print("【意图缺省】无意图缺省，意图完整！")
            return question

        # 获取最近若干条历史（user/assistant）
        recent = []
        messages = []

        if LANGCHAIN_AVAILABLE and hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
        elif hasattr(self.memory, 'buffer'):
            messages = self.memory.buffer

        cnt = 6    # 获取最近3段聊天记录
        print(f"需要改写的问题前记忆模块的对话条数为：{len(messages)}")
        recent_msgs = messages[-cnt:] if len(messages) > 0 else []
        # 保留用户问题和大模型回复
        for m in recent_msgs:
            if LANGCHAIN_AVAILABLE and hasattr(m, 'type'):
                role = '用户' if m.type == 'human' else '助手'
                content = m.content
            elif isinstance(m, dict):
                role = '用户' if m.get('role') == 'user' else '助手'
                content = m.get('content', '')
            else:
                continue
            recent.append(f"{role}: {content}")
        # # 只保留用户问题
        # for m in recent_msgs:
        #     if LANGCHAIN_AVAILABLE and hasattr(m, 'type'):
        #         if m.type != 'human':  # 只保留用户
        #             continue
        #         role = '用户'
        #         content = m.content
        #     elif isinstance(m, dict):
        #         if m.get('role') != 'user':  # 只保留用户
        #             continue
        #         role = '用户'
        #         content = m.get('content', '')
        #     else:
        #         continue
        #     recent.append(f"{role}: {content}")     ，与被引用的历史问题的句子结构一致
        recent_text = "\n".join(recent)

        prompt = (
            "你是一个问句改写器。用户的当前问题有省略或者有代词引用用户最近的历史问题，请根据用户最近的历史对话，将它改写为一个独立、完整的问句，要求必须包含主语、谓语、宾语结构。"
            "并保留用户原意。只输出改写后的问句，不要输出其它解释。\n\n"
            f"用户最近的历史对话：\n{recent_text}\n\n"
            f"当前问题：{question}\n\n改写后的问题："
        )
        try:
            settings = DeepSeekSettings.from_env()
            client = create_deepseek_client(settings)
            completion = client.chat.completions.create(
                model=settings.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512
            )
            rewritten = completion.choices[0].message.content.strip().strip('"').strip()
            # 去除<think>标签及其内容
            import re
            rewritten = re.sub(r'<think>.*?</think>', '', rewritten, flags=re.DOTALL).strip()
            # 若返回过长/不合理，fallback
            if not rewritten:
                return question
            if len(rewritten) > 1024:
                return question
            return rewritten
        except Exception as e:
            # fallback：将最近的最后一条 user 消息拼在前面（非常简单）
            last_user = None
            messages = []
            if LANGCHAIN_AVAILABLE and hasattr(self.memory, 'chat_memory'):
                messages = self.memory.chat_memory.messages
            elif hasattr(self.memory, 'buffer'):
                messages = self.memory.buffer

            for msg in reversed(messages):
                role = msg.get('role') if isinstance(msg, dict) else getattr(msg, 'type', None)
                content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                if role == 'user' or role == 'human':
                    last_user = content
                    break
            if last_user:
                candidate = f"{last_user}；现在用户问：{question}"
                return candidate
            return question

    def detect_intent_inheritance(self, question: str, similarity_threshold: float = 0.85,
                                  consider_last_user: bool = True) -> Dict[str, Any]:
        """
        判断当前问题是否继承了上一次检索/意图。
        返回 dict：{"inherit": bool, "reason": str, "reused_chunks": List[chunk] or None}
        思路：
         - 若最近一次用户问题与当前问题 embedding 相似度高，则视为继承
         - 或当前问题包含明显代词/时间，可比对与上次检索结果的相似度
        """
        result = {"inherit": False, "reason": "", "reused_chunks": None, "similarity": 0.0}
        try:
            # 1) 获取上一次的用户问题 last_user（若有）
            last_user = None
            messages = []
            if LANGCHAIN_AVAILABLE and hasattr(self.memory, 'chat_memory'):
                messages = self.memory.chat_memory.messages
            elif hasattr(self.memory, 'buffer'):
                messages = self.memory.buffer

            # 从后往前找第一个 role user
            for msg in reversed(messages):
                role = msg.get('role') if isinstance(msg, dict) else getattr(msg, 'type', None)
                content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                if role == 'user' or role == 'human':
                    last_user = content
                    break
            # last_user：用户上一次的问题  question：当前问题
            # print(f"进入detect_intent_inheritance意图函数！\n")
            print(f"用户上一次的问题为：{last_user}")
            q_emb = self.embedding_model.embed_text(question)
            if consider_last_user and last_user:
                last_emb = self.embedding_model.embed_text(last_user)
                # 计算余弦相似度
                qv = np.array(q_emb)
                lv = np.array(last_emb)
                sim = float(qv @ lv / (norm(qv) * norm(lv) + 1e-8))   # 用户上一次问题与当前问题的余弦相似度 sim
                print(f"用户上一次的问题与当前问题的余弦相似度为：{sim:.3f}")
                if sim >= similarity_threshold:
                    result["inherit"] = True
                    result["reason"] = f"与上次用户问题相似(sim={sim:.3f})"
                    result["similarity"] = sim
                    # 若上次有检索结果则复用
                    # 尝试从内存搜寻最近一次检索chunk（若你保存了到 memory 或其他位置）
                    if hasattr(self, 'last_retrieved_chunks') and self.last_retrieved_chunks:
                        print(f"直接复用上一轮检索到的{len(self.last_retrieved_chunks)}个chunk\n")
                        result["reused_chunks"] = self.last_retrieved_chunks
                    return result

        except Exception as e:
            # 出错不阻塞主流程，返回不继承
            result["reason"] = f"检测失败: {str(e)}"
            return result

        result["reason"] = "未检测到明显继承"
        return result


    # def detect_intent_inheritance(
    #         self,
    #         question: str,
    #         similarity_threshold: float = 0.80,
    #         keyword_threshold: float = 0.80,
    #         weight_sentence: float = 0.4,
    #         weight_keyword: float = 0.6
    # ) -> Dict[str, Any]:
    #     """
    #     融合整句向量与关键词（主语/谓语/宾语）向量的意图继承检测
    #     """
    #     result = {
    #         "inherit": False,
    #         "reason": "",
    #         "reused_chunks": None,
    #         "similarity": 0.0
    #     }
    #
    #     try:
    #         # 1. 获取上一次用户问题
    #         last_user = None
    #         messages = []
    #         if LANGCHAIN_AVAILABLE and hasattr(self.memory, 'chat_memory'):
    #             messages = self.memory.chat_memory.messages
    #         elif hasattr(self.memory, 'buffer'):
    #             messages = self.memory.buffer
    #
    #         for msg in reversed(messages):
    #             role = msg.get('role') if isinstance(msg, dict) else getattr(msg, 'type', None)
    #             content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
    #             if role in ('user', 'human'):
    #                 last_user = content
    #                 break
    #
    #         if not last_user:
    #             result["reason"] = "无历史问题"
    #             return result
    #
    #         # 2. 整句向量相似度
    #         self.embedding_model = EmbeddingModel(batch_size=32)
    #         q_emb = self.embedding_model.embed_text(question)
    #         last_emb = self.embedding_model.embed_text(last_user)
    #         qv = np.array(q_emb)
    #         lv = np.array(last_emb)
    #         sim_sentence = float(dot(qv, lv) / (norm(qv) * norm(lv) + 1e-8))
    #
    #         # 3. 直接判定
    #         if sim_sentence >= similarity_threshold:
    #             result["inherit"] = True
    #             result["similarity"] = sim_sentence
    #             result["reason"] = f"融合相似度={sim_sentence:.3f}"
    #             if hasattr(self, 'last_retrieved_chunks') and self.last_retrieved_chunks:
    #                 result["reused_chunks"] = self.last_retrieved_chunks
    #             print(f"【意图继承】整句相似度={sim_sentence:.3f}")
    #         else:
    #             print(f"【意图继承失败】整句相似度={sim_sentence:.3f}")
    #
    #     except Exception as e:
    #         result["reason"] = f"检测失败: {str(e)}"
    #
    #     return result


    def generate_answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        检索+大模型生成式回答，返回结构化结果（支持历史对话记忆）
        """
        # qwen_api_key = os.getenv('LOCAL_API_KEY')
        # qwen_base_url = os.getenv('LOCAL_BASE_URL')
        # qwen_model = os.getenv('LOCAL_TEXT_MODEL')
        # client = OpenAI(api_key=qwen_api_key, base_url=qwen_base_url)

        settings = DeepSeekSettings.from_env()
        client = create_deepseek_client(settings)

        # # 1. 省略解析 / 问题改写
        # rewritten_question = self.rewrite_question_with_context(question)
        # # 将改写动作记录到memory（方便可解释性）
        # print(f"【问题改写】{question} -> {rewritten_question}")

        # 1. 意图继承检测
        inheritance = self.detect_intent_inheritance(question)
        if inheritance.get("inherit") and inheritance.get("reused_chunks"):
            # 直接复用上一轮检索结果
            reranked_chunks = inheritance["reused_chunks"]
            print(f"【意图继承】，直接复用 {len(reranked_chunks)} 个 chunk，跳过检索与重排")
        else:
            print(f"【意图继承】失败，无法继承上次的reranked_chunks，继续往下执行")
            # 2. 省略解析 / 问题改写
            rewritten_question = self.rewrite_question_with_context(question)
            # 将改写动作记录到memory（方便可解释性）
            print(f"【问题改写】{question} -> {rewritten_question}")

            # 原来的混合检索 + 重排流程
            # retrieved_chunks = self.search_hybrid(rewritten_question)
            # 替换为实体级联检索
            retrieved_chunks = self.search_entity_cascade(rewritten_question)

            if not retrieved_chunks:
                return {
                    "question": question,
                    "answer": "",
                    "filename": "",
                    "page": "",
                    "retrieval_chunks": []
                }

            pairs = [[rewritten_question, c["content"]] for c in retrieved_chunks]
            scores = self.reranker.compute_score(pairs)
            for i, sc in enumerate(scores):
                retrieved_chunks[i]["score"] = sc
            reranked_chunks = sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True)[:top_k]
        print(f"   重排后检索到 {len(reranked_chunks)} 个结果")


        # # 2. 意图继承检测（决定是否重用上次检索结果）
        # inheritance = self.detect_intent_inheritance(rewritten_question)
        # reuse_chunks = None
        # if inheritance.get("inherit") and inheritance.get("reused_chunks"):
        #     reuse_chunks = inheritance.get("reused_chunks")
        #
        # # 混合检索，粗召回
        # retrieved_chunks = self.search_hybrid(rewritten_question)
        #
        # if not retrieved_chunks:
        #     return {
        #         "question": question,
        #         "answer": "",
        #         "filename": "",
        #         "page": "",
        #         "retrieval_chunks": []
        #     }
        #
        # # 用 FlagReranker 精排
        # pairs = [[rewritten_question, chunk['content']] for chunk in retrieved_chunks]
        # scores = self.reranker.compute_score(pairs)  # 返回每个pair的相关性分数
        # print(scores)
        # # 绑定分数
        # for i, sc in enumerate(scores):
        #     retrieved_chunks[i]['score'] = sc
        # # 按分数排序，取 top_k
        # reranked_chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)[:top_k]


        # 新增：把本轮检索结果缓存下来，供下一轮复用
        self.last_retrieved_chunks = reranked_chunks
        # 拼接检索内容，带上元数据
        context = "\n".join([
            f"[文件名]{c['metadata']['file_name']} [页码]{c['metadata']['page']}\n{c['content']}" for c in reranked_chunks
        ])

        # 获取历史对话
        history_messages = []
        if hasattr(self.memory, 'buffer') and self.memory.buffer:
            if LANGCHAIN_AVAILABLE:
                # Langchain格式 - 获取chat_memory中的消息
                for msg in self.memory.chat_memory.messages:
                    if hasattr(msg, 'type') and hasattr(msg, 'content'):
                        role = 'user' if msg.type == 'human' else 'assistant'
                        history_messages.append({"role": role, "content": msg.content})
            else:
                # 简单格式
                for msg in self.memory.buffer:
                    history_messages.append({"role": msg.get('role', 'user'), "content": msg.get('content', '')})

        # 构建包含历史对话的消息
        messages = [
            {"role": "system", "content": "你是一名专业的金融分析助手，能够记住之前的对话内容，为用户提供连贯的金融分析服务。"}
        ]

        # 添加历史消息
        messages.extend(history_messages)

        # 添加当前问题和上下文
        prompt = (
            f"你是一名专业的金融分析助手，根据以下检索到的内容回答用户当前问题，同时参考之前的对话历史。\n"
            f"请严格按照如下JSON格式输出：\n"
            f'{{"answer": "你的简洁回答", "filename": "文件名.pdf", "page": "来源页码"}}'"\n"
            f"请必须按照上面的JSON格式输出，确保输出内容为合法JSON字符串，不要输出多余内容或者别的格式的JSON字符串。\n"
            f"检索内容：\n{context}\n\n当前问题：{question}"
        )
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )
        import json as pyjson
        from extract_json_array import extract_json_array
        raw = completion.choices[0].message.content.strip()
        import re
        # 1. 先把 <think> 内容抓出来当 reasoning
        think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ''
        # 2. 再把 <think> 标签整体去掉，剩下的当正文
        body = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = body
        # print("模型原始回复：", raw)
        # 用 extract_json_array 提取 JSON 对象
        json_str = extract_json_array(raw, mode='objects')
        if json_str:
            try:
                print("   原始JSON字符串:", json_str)
                arr = pyjson.loads(json_str)
                # 只取第一个对象
                if isinstance(arr, list) and arr:
                    j = arr[0]
                    print("   提取到的JSON对象:", j)
                    answer = j.get('answer', '')
                    filename = j.get('filename', '')
                    page = j.get('page', '')
                else:
                    print("   未提取到合法 JSON，走兜底分支")
                    answer = raw
                    filename = reranked_chunks[0]['metadata']['file_name'] if reranked_chunks else ''
                    page = reranked_chunks[0]['metadata']['page'] if reranked_chunks else ''
            except Exception:
                print("   未提取到")
                answer = raw
                filename = reranked_chunks[0]['metadata']['file_name'] if reranked_chunks else ''
                page = reranked_chunks[0]['metadata']['page'] if reranked_chunks else ''
        else:
            answer = raw
            filename = reranked_chunks[0]['metadata']['file_name'] if reranked_chunks else ''
            page = reranked_chunks[0]['metadata']['page'] if reranked_chunks else ''

        # ========= 保存 trace =========
        self.latest_trace = self.feedback._pack_reasoning_trace(
            question, reranked_chunks, reasoning, answer
        )
        # print("   最新trace:", self.latest_trace)

        # # 简化解析：直接 json.loads，失败就兜底
        # try:
        #     data = pyjson.loads(raw)
        #     answer = data.get("answer", "")
        #     filename = data.get("filename", "")
        #     page = str(data.get("page", ""))
        # except Exception:
        #     answer = raw
        #     filename = reranked_chunks[0]['metadata']['file_name'] if reranked_chunks else ''
        #     page = str(reranked_chunks[0]['metadata']['page']) if reranked_chunks else ''

        # 保存到对话记忆（使用滑动窗口记忆机制）
        self.add_conversation_pair(question, answer)

        # 结构化输出
        return {
            "question": question,
            "answer": answer,
            "filename": filename,
            "page": page,
            "retrieval_chunks": reranked_chunks
        }


if __name__ == '__main__':
    # 路径可根据实际情况调整
    chunk_json_path = "./all_pdf_page_chunks.json"
    rag = SimpleRAG(chunk_json_path)
    rag.setup()

    # 支持用户对话输入
    print("RAG系统已启动，支持历史对话记忆功能！")
    print("可用命令：")
    print("  - 直接输入问题开始对话")
    print("  - 'quit' 退出程序")
    print("  - 'clear' 清除当前对话记忆")
    print("  - 'history' 查看对话历史")
    print("  - 'save' 保存对话历史到文件")
    print("  - 'load' 从文件加载对话历史")

    # 启动时尝试加载历史对话
    rag.load_conversation_history()

    while True:
        # question = input("\n\n请输入问题或命令: ").strip()
        import sys
        print("\n\n请输入问题或命令: ", end='', flush=True)
        question = sys.stdin.buffer.readline().decode('utf-8', errors='ignore').strip()
        if question.lower() in ['quit', 'exit', '退出']:
            # 退出前保存对话历史
            rag.save_conversation_history()
            print("感谢使用，再见！")
            break
        elif question.lower() == 'clear':
            # 清除对话记忆，创建新的空记忆实例
            rag.memory = ConversationBufferMemory(return_messages=True)
            print("对话记忆已清除！")
            continue
        elif question.lower() == 'history':
            # 准备要显示的消息列表
            messages_to_show = []
            if LANGCHAIN_AVAILABLE and hasattr(rag.memory, 'chat_memory'):
                messages_to_show = rag.memory.chat_memory.messages
            elif hasattr(rag.memory, 'buffer'):
                messages_to_show = rag.memory.buffer

            # 显示对话历史
            if messages_to_show:
                print("\n=== 对话历史 ===")
                for i, msg in enumerate(messages_to_show, 1):
                    if LANGCHAIN_AVAILABLE and hasattr(msg, 'type'):
                        # 使用Langchain格式显示消息
                        role = "用户" if msg.type == "human" else "助手"
                        print(f"{i}. {role}: {msg.content[:100]}...")
                    else:
                        # 使用自定义格式显示消息
                        role = "用户" if msg.get('role') == 'user' else "助手"
                        content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
                        print(f"{i}. {role}: {content[:100]}...")
            else:
                print("暂无对话历史")
            continue
        elif question.lower() == 'save':
            # 保存对话历史到文件
            rag.save_conversation_history()
            continue
        elif question.lower() == 'load':
            # 从文件加载对话历史
            rag.load_conversation_history()
            continue
        elif question.lower().startswith("feedback"):
            try:
                _, rest = question.split(" ", 1)
                q, fb = rest.split("|||", 1)
                if rag.latest_trace is None:
                    print("暂无推理链，无法回溯")
                    continue
                new = rag.feedback.regenerate_with_loop(
                    q, rag.latest_trace, fb, max_loop=3
                )
                print("\n【修正后回答】")
                print(new["answer"])
                if new["filename"]:
                    print(f"来源: {new['filename']} 第{new['page']}页")
                # 更新 trace
                rag.latest_trace.update({
                    "reasoning": new["reasoning"],
                    "draft": new["answer"]
                })
                continue
            except Exception as e:
                print("格式应为：feedback 原问题|||用户反馈")
                continue

        if not question:
            print("请输入有效的问题！")
            continue

        try:
            # 生成并显示回答
            result = rag.generate_answer(question, top_k=5)
            print(f"\n回答: {result['answer']}")
            if result['filename'] and result['page']:
                print(f"来源: {result['filename']} 第{result['page']}页")
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")
