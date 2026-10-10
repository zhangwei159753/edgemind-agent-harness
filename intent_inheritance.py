from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
LANGCHAIN_AVAILABLE = True
load_dotenv()
import hanlp
import numpy as np
from numpy import dot
from numpy.linalg import norm
from get_text_embedding import get_text_embedding
import os

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

# --------------------------------------------------
# 1. 统一 pipeline：先分词，再依存
# --------------------------------------------------
pipeline = (
    hanlp.pipeline()
    .append(hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH))
    .append(hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL))
)


def _extract_keywords(text: str) -> str:
    """
    基于 HanLP 2.x CoNLL-U 对象提取【主语短语】【宾语短语】【时间状语短语】
    找不到则返回空串
    """
    try:
        conll = pipeline(text)          # 返回 CoNLL-U 对象
        words = [tok.form for tok in conll]
        heads = [tok.head for tok in conll]
        rels  = [tok.deprel for tok in conll]

        # 收集以 idx 为根的全部子节点（含自身）
        def subtree(root_idx):
            idxs = [root_idx]
            queue = [root_idx]
            while queue:
                cur = queue.pop()
                for i, h in enumerate(heads):
                    if h - 1 == cur and i != cur:
                        idxs.append(i)
                        queue.append(i)
            return sorted(idxs)

        # 不再用 subtree，直接取 token 自身
        subj_idxs, obj_idxs, time_idxs, assmod_idxs, nn_idxs, top_idxs = [], [], [], [], [], []
        for i, (h, r) in enumerate(zip(heads, rels)):
            if r == 'nsubj':    # 主语
                subj_idxs.append(i)          # 仅当前词
            elif r in ('obj', 'dobj', 'iobj'):    # 宾语
                obj_idxs.append(i)
            elif r in ('obl:tmod', 'pobj'):    # 时间状语
                time_idxs = [i]
            elif r == 'assmod':    # 同位修饰语  “在”
                assmod_idxs = [i]
            elif r == 'nn':    # 名词
                nn_idxs = [i]
            elif r == 'top':    # 主题
                top_idxs = [i]

        subj = ' '.join(words[i] for i in subj_idxs) if subj_idxs else ''
        obj  = ' '.join(words[i] for i in obj_idxs)  if obj_idxs  else ''
        time = ' '.join(words[i] for i in time_idxs) if time_idxs else ''
        assmod = ' '.join(words[i] for i in assmod_idxs) if assmod_idxs else ''
        nn = ' '.join(words[i] for i in nn_idxs) if nn_idxs else ''
        top = ' '.join(words[i] for i in top_idxs) if top_idxs else ''

        # 拼成一句话返回
        parts = [subj, obj, time, assmod, nn, top]
        # 意图缺省判断：主语/宾语/时间任一缺失
        missing = (not subj) or (not obj) or (not time)
        # 收集缺失成分名称
        missing_parts = []
        if not subj:
            missing_parts.append("主语")
        if not obj:
            missing_parts.append("宾语")
        if not time:
            missing_parts.append("状语")
        return ' '.join(p for p in parts if p), missing, missing_parts
        # return {
        #     "subject": subj,
        #     "object": obj,
        #     "time": time,
        #     "dobj": dobj,
        #     "assmod": assmod,
        #     "nn": nn,
        #     "top": top
        # }

    except Exception:
        return ""


def extract_keywords_with_intent_check(text: str):
    """
    返回 (keywords:str, intent_missing:bool)
    任一关键槽位（主语/宾语/时间）缺失即 intent_missing=True
    """
    keywords, missing, missing_parts = _extract_keywords(text)
    return keywords, missing, missing_parts

# --------------------------------------------------
# 2. 意图继承检测（保持对外接口不变）
# --------------------------------------------------
def detect_intent_inheritance(
        self,
        question: str,
        similarity_threshold: float = 0.75,
        keyword_threshold: float = 0.80,
        weight_sentence: float = 0.4,
        weight_keyword: float = 0.6
) -> Dict[str, Any]:
    """
    融合整句向量与关键词（主语/谓语/宾语）向量的意图继承检测
    """
    result = {
        "inherit": False,
        "reason": "",
        "reused_chunks": None,
        "similarity": 0.0
    }

    try:
        # 1. 获取上一次用户问题
        last_user = None
        messages = []
        if LANGCHAIN_AVAILABLE and hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
        elif hasattr(self.memory, 'buffer'):
            messages = self.memory.buffer

        for msg in reversed(messages):
            role = msg.get('role') if isinstance(msg, dict) else getattr(msg, 'type', None)
            content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
            if role in ('user', 'human'):
                last_user = content
                break

        if not last_user:
            result["reason"] = "无历史问题"
            return result

        # 2. 整句向量相似度
        # self.embedding_model = EmbeddingModel(batch_size=32)
        q_emb = self.embedding_model.embed_text(question)
        last_emb = self.embedding_model.embed_text(last_user)
        qv = np.array(q_emb)
        lv = np.array(last_emb)
        sim_sentence = float(dot(qv, lv) / (norm(qv) * norm(lv) + 1e-8))

        # 3. 关键词提取 + 关键词向量相似度
        kw_cur = _extract_keywords(question)
        kw_last = _extract_keywords(last_user)
        if not kw_cur or not kw_last:
            sim_keyword = 0.0
        else:
            kcv, klv = (
                np.array(self.embedding_model.embed_text(kw_cur)),
                np.array(self.embedding_model.embed_text(kw_last)),
            )
            sim_keyword = float(dot(kcv, klv) / (norm(kcv) * norm(klv) + 1e-8))

        # 4. 融合得分
        final_sim = weight_sentence * sim_sentence + weight_keyword * sim_keyword

        if final_sim >= similarity_threshold:
            result["inherit"] = True
            result["similarity"] = final_sim
            result["reason"] = f"融合相似度={final_sim:.3f}"
            if hasattr(self, 'last_retrieved_chunks') and self.last_retrieved_chunks:
                result["reused_chunks"] = self.last_retrieved_chunks
            print(
                f"【意图继承】整句相似度={sim_sentence:.3f}, 关键词相似度={sim_keyword:.3f}, 融合得分={final_sim:.3f}")
        else:
            print(
                f"【意图继承失败】整句相似度={sim_sentence:.3f}, 关键词相似度={sim_keyword:.3f}, 融合得分={final_sim:.3f}")

    except Exception as e:
        result["reason"] = f"检测失败: {str(e)}"

    return result


class MockMemory:
    """极简内存 mock，仅用于测试"""
    def __init__(self, history: list):
        self.chat_memory = type('', (), {'messages': history})()

class MockEmbedding:
    def embed_text(self, text):
        # 简单随机向量，仅演示
        rng = np.random.default_rng(abs(hash(text)) % 2**32)
        return rng.random(128).tolist()

class MockRAG:
    def __init__(self, history=None):
        self.memory = MockMemory(history or [])
        self.embedding_model = EmbeddingModel()
        self.last_retrieved_chunks = []

def test_intent_inheritance():
    rag = MockRAG([
        {"role": "user", "content": "千味央厨公司在2021年推出了哪些新产品"},
        {"role": "assistant", "content": "推出了小笼包、油条等。"}
    ])

    new_q = "千味央厨公司的新产品在制作和销售上各自都有什么特点"  # 千味央厨公司的新产品在制作和销售上各自都有什么特点
    ret = detect_intent_inheritance(rag, new_q, similarity_threshold=0.75)
    print("测试新问：", new_q)
    print("继承结果：", ret)


# if __name__ == "__main__":
#     test_intent_inheritance()

if __name__ == "__main__":

    # test_intent_inheritance()

    test_sentences = [
        "未来前景如何",  # 只有形容词或属性，缺少被修饰对象（谁的毛利率）
        "比2022年增加了多少",  # 只有谓语或动词，没有主语、宾语或时间限定（提高了多少收入）
        "在华东地区表现出了哪些效果？",  # 仅出现时间或地点，缺少主体
        "广联达公司如何通过云转型推动其数字施工业务的发展？",
        "千味央厨公司在2021年推出了哪些新产品",
        "千味央厨公司推出的这些新产品在制作上都有什么特点",
        "伊利股份公司在2024年的营收是多少？",
        "它去年的净利润是多少？",
        "张三投资了比亚迪的股票。",
    ]

    def show_svo(s: str):
        print(f"\n原句：{s}")
        keywords, missing, missing_parts = _extract_keywords(s)
        if missing:
            print("提取：", keywords, "意图缺省", f"（缺失：{missing_parts}）")
        else:
            print("提取：", keywords, "意图完整！")


    for sent in test_sentences:
        show_svo(sent)
