from __future__ import annotations

import os
import re
from typing import Any

from .integrations.fetch import SafePageFetcher
from .integrations.search import BraveSearchClient
from .integrations.weather import OpenMeteoWeatherClient


class BuiltinToolRegistry:
    def __init__(self, weather: OpenMeteoWeatherClient | None = None,
                 search: BraveSearchClient | None = None,
                 fetcher: SafePageFetcher | None = None):
        self.weather = weather or OpenMeteoWeatherClient()
        self.search = search or BraveSearchClient()
        self.fetcher = fetcher or SafePageFetcher()

    @staticmethod
    def _brave_ready() -> bool:
        return bool(os.getenv("BRAVE_SEARCH_API_KEY", "").strip())

    def model_tools(self) -> list[dict]:
        result = [{
            "type": "function",
            "function": {
                "name": "edgemind__get_weather",
                "description": "查询指定城市的当前天气或未来预报；地点不明确时会返回候选地区。",
                "parameters": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "location": {"type": "string", "minLength": 1, "maxLength": 160},
                        "mode": {"type": "string", "enum": ["current", "forecast"]},
                        "forecast_days": {"type": "integer", "minimum": 1, "maximum": 7},
                    },
                    "required": ["location"],
                },
            },
        }]
        if self._brave_ready():
            result.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "edgemind__web_search",
                        "description": "通过 Brave Search 搜索网页，并返回带标题、摘要、链接和时间的来源。",
                        "parameters": {
                            "type": "object", "additionalProperties": False,
                            "properties": {
                                "query": {"type": "string", "minLength": 1, "maxLength": 600},
                                "count": {"type": "integer", "minimum": 1, "maximum": 5},
                                "country": {"type": "string", "pattern": "^[A-Za-z]{2}$"},
                                "search_lang": {"type": "string", "minLength": 2, "maxLength": 16},
                                "freshness": {"type": ["string", "null"],
                                              "enum": ["pd", "pw", "pm", "py", None]},
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "edgemind__fetch_search_result",
                        "description": "读取本轮网页搜索结果或用户提供的公开 HTML/纯文本页面。",
                        "parameters": {
                            "type": "object", "additionalProperties": False,
                            "properties": {"url": {"type": "string", "minLength": 1, "maxLength": 2048}},
                            "required": ["url"],
                        },
                    },
                },
            ])
        return result

    def status(self) -> list[dict]:
        ready = self._brave_ready()
        return [
            {"id": "open_meteo", "label": "Open-Meteo 天气", "ready": True,
             "detail": "天气与地点查询可用；个人非商业使用。"},
            {"id": "brave_search", "label": "Brave Search", "ready": ready,
             "detail": "已配置。" if ready else "未配置；请在本机 .env 设置 BRAVE_SEARCH_API_KEY。"},
        ]

    def execute(self, function_name: str, arguments: dict,
                allowed_urls: set[str] | None = None) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        if function_name == "edgemind__get_weather":
            self._only(arguments, {"location", "mode", "forecast_days"})
            location = arguments.get("location")
            if not isinstance(location, str) or not location.strip() or len(location) > 160:
                raise ValueError("location 必须是 1–160 个字符")
            mode = arguments.get("mode", "current")
            days = arguments.get("forecast_days", 1)
            if mode not in {"current", "forecast"}:
                raise ValueError("mode must be current or forecast")
            if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 7:
                raise ValueError("forecast_days must be between 1 and 7")
            return self.weather.get_weather(location, mode, days)
        if function_name == "edgemind__web_search":
            self._only(arguments, {"query", "count", "country", "search_lang", "freshness"})
            if not self._brave_ready():
                from .integrations.search import SearchServiceError
                raise SearchServiceError("not_configured", "网页搜索未配置 BRAVE_SEARCH_API_KEY。")
            query = arguments.get("query")
            count = arguments.get("count", 5)
            country = arguments.get("country", "CN")
            search_lang = arguments.get("search_lang", "zh-hans")
            freshness = arguments.get("freshness")
            if not isinstance(query, str) or not query.strip() or len(query) > 600:
                raise ValueError("query 必须是 1–600 个字符")
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
                raise ValueError("count must be between 1 and 5")
            if not isinstance(country, str) or not re.fullmatch(r"[A-Za-z]{2}", country):
                raise ValueError("country must be a two-letter country code")
            if not isinstance(search_lang, str) or len(search_lang) > 16:
                raise ValueError("search_lang must be a short language code")
            if freshness is not None and freshness not in {"pd", "pw", "pm", "py"}:
                raise ValueError("freshness must be pd, pw, pm, py, or null")
            return self.search.search(query, count, country, search_lang, freshness)
        if function_name == "edgemind__fetch_search_result":
            self._only(arguments, {"url"})
            url = arguments.get("url")
            if not isinstance(url, str) or not url or len(url) > 2048:
                raise ValueError("url 必须是 1–2048 个字符")
            if not self._brave_ready():
                raise ValueError("网页读取需要先配置 Brave Search API key")
            return self.fetcher.fetch(url, allowed_urls or set())
        raise ValueError(f"未注册的内置工具：{function_name}")

    @staticmethod
    def _only(arguments: dict, allowed: set[str]) -> None:
        unexpected = set(arguments) - allowed
        if unexpected:
            raise ValueError("工具参数包含未允许的字段")
