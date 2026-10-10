from http import client
from openai import OpenAI
from dotenv import load_dotenv
import os
import hashlib

from typing import List, Dict, Optional, Tuple
import json

# LOCAL_API_KEY,LOCAL_BASE_URL,LOCAL_TEXT_MODEL,LOCAL_EMBEDDING_MODEL

load_dotenv()  # 加载环境变量（可选，用户可自行读取）



def get_openai_client(api_key: str, base_url: str) -> OpenAI:
    """
    获取 OpenAI 客户端，必须传递 api_key 和 base_url
    """
    if not api_key or not base_url:
        raise ValueError("api_key 和 base_url 必须显式传递！")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )



from tqdm import tqdm

def batch_get_embeddings(
    texts: List[str],
    batch_size: int = 64,
    api_key: str = None,
    base_url: str = None,
    embedding_model: str = None
) -> List[List[float]]:
    """
    批量获取文本的嵌入向量
    :param texts: 文本列表
    :param batch_size: 批处理大小
    :param api_key: 可选，自定义 API KEY
    :param base_url: 可选，自定义 BASE URL
    :param embedding_model: 可选，自定义嵌入模型
    :return: 嵌入向量列表
    """
    if not api_key or not base_url or not embedding_model:
        raise ValueError("api_key、base_url、embedding_model 必须显式传递！")
    all_embeddings = []
    client = get_openai_client(api_key, base_url)
    total = len(texts)
    if total == 0:
        return []
    iterator = range(0, total, batch_size)
    if total > 1:
        iterator = tqdm(iterator, desc="Embedding", unit="batch")
    import time
    from openai import RateLimitError
    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        retry_count = 0
        while True:
            try:
                response = client.embeddings.create(
                    model=embedding_model,
                    input=batch_texts
                )
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except RateLimitError as e:
                retry_count += 1
                print(f"RateLimitError: {e}. 等待10秒后重试（第{retry_count}次）...")
                time.sleep(10)
    return all_embeddings



def get_text_embedding(
    texts: List[str],
    api_key: str = None,
    base_url: str = None,
    embedding_model: str = None,
    batch_size: int = 64
) -> List[List[float]]:
    """
    获取文本的嵌入向量，支持批次处理，保持输出顺序与输入顺序一致
    :param texts: 文本列表
    :param api_key: 可选，自定义 API KEY
    :param base_url: 可选，自定义 BASE URL
    :param embedding_model: 可选，自定义嵌入模型
    :param batch_size: 批处理大小
    :return: 嵌入向量列表
    """
    if not api_key or not base_url or not embedding_model:
        raise ValueError("api_key、base_url、embedding_model 必须显式传递！")
    # 直接批量获取所有文本的embedding，不做缓存
    return batch_get_embeddings(
        texts,
        batch_size=batch_size,
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model
    )
