from __future__ import annotations

import os
import re
from urllib.parse import urlparse
from typing import Any

import httpx


class SearchServiceError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class BraveSearchClient:
    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None,
                 timeout_s: float = 8.0):
        self._api_key = (api_key if api_key is not None else os.getenv(
            "BRAVE_SEARCH_API_KEY", "")).strip()
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
        self._timeout_s = timeout_s

    def search(self, query: str, count: int = 5, country: str = "CN",
               search_lang: str = "zh-hans", freshness: str | None = None) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise ValueError("搜索内容不能为空")
        if len(query) > 600:
            raise ValueError("搜索内容最多 600 个字符")
        if len(query.split()) > 75:
            raise ValueError("搜索词最多 75 个")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
            raise ValueError("count must be between 1 and 5")
        country = (country or "").upper()
        if not re.fullmatch(r"[A-Z]{2}", country):
            raise ValueError("country must be a two-letter country code")
        if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", search_lang or ""):
            raise ValueError("search_lang must be a valid language code")
        if freshness is not None and freshness not in {"pd", "pw", "pm", "py"} and not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}", freshness
        ):
            raise ValueError("freshness must be pd, pw, pm, py, or a valid date range")
        if not self._api_key:
            raise SearchServiceError("not_configured", "网页搜索未配置 BRAVE_SEARCH_API_KEY。")

        params = {"q": query, "count": count, "country": country,
                  "search_lang": search_lang, "safesearch": "moderate"}
        if freshness:
            params["freshness"] = freshness
        try:
            response = self._client.get(
                self.API_URL, params=params,
                headers={"Accept": "application/json", "X-Subscription-Token": self._api_key},
                timeout=self._timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise SearchServiceError("timeout", "网页搜索请求超时，请稍后重试。") from exc
        except httpx.HTTPError as exc:
            raise SearchServiceError("network_error", "无法连接网页搜索服务，请检查网络后重试。") from exc
        if response.status_code == 401 or response.status_code == 403:
            raise SearchServiceError("authentication_failed", "Brave Search API key 无效或无权访问。")
        if response.status_code == 429:
            raise SearchServiceError("rate_limited", "Brave Search 请求额度或速率受限，请稍后重试。")
        if response.status_code >= 400:
            raise SearchServiceError("upstream_error", f"Brave Search 返回 HTTP {response.status_code}。")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SearchServiceError("invalid_response", "Brave Search 返回了无效数据。") from exc
        if not isinstance(payload, dict):
            raise SearchServiceError("invalid_response", "Brave Search 返回了无效数据。")

        web = payload.get("web") or {}
        normalized = []
        for item in web.get("results", [])[:count]:
            url = item.get("url")
            if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"}:
                continue
            normalized.append({
                "title": str(item.get("title") or "")[:500],
                "url": url[:2048],
                "snippet": str(item.get("description") or "")[:2000],
                "published_at": item.get("page_age") or item.get("page_fetched"),
                "source": urlparse(url).hostname or "",
            })
        return {"status": "succeeded", "query": query, "results": normalized,
                "result_count": len(normalized), "source": "Brave Search"}
