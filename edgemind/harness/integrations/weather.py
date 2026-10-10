from __future__ import annotations

import copy
import time
from typing import Any

import httpx


class WeatherServiceError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class OpenMeteoWeatherClient:
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    ATTRIBUTION = "Weather data by Open-Meteo; geocoding data by GeoNames."

    def __init__(self, client: httpx.Client | None = None, timeout_s: float = 8.0,
                 cache_ttl_s: float = 300.0, clock=time.monotonic):
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
        self._timeout_s = timeout_s
        self._cache_ttl_s = cache_ttl_s
        self._clock = clock
        self._cache: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}

    def get_weather(self, location: str, mode: str = "current",
                    forecast_days: int = 1) -> dict[str, Any]:
        location = (location or "").strip()
        if not location:
            raise ValueError("请提供城市或地区名称")
        if mode not in {"current", "forecast"}:
            raise ValueError("mode must be current or forecast")
        if isinstance(forecast_days, bool) or not isinstance(forecast_days, int) or not 1 <= forecast_days <= 7:
            raise ValueError("forecast_days must be between 1 and 7")

        cache_key = (location.casefold(), mode, forecast_days)
        cached = self._cache.get(cache_key)
        if cached and cached[0] > self._clock():
            return copy.deepcopy(cached[1])

        geocoding = self._get_json(self.GEOCODING_URL, {
            "name": location, "count": 5, "language": "zh", "format": "json"
        })
        candidates = geocoding.get("results") or []
        if not candidates:
            return {"status": "not_found", "query": location,
                    "message": f"没有找到地点“{location}”，请补充国家或省/州名称。"}
        if len(candidates) > 1:
            return {"status": "needs_clarification", "query": location,
                    "location_candidates": [self._candidate(item) for item in candidates[:5]],
                    "message": "找到多个同名或近似地点，请指定国家或省/州。"}

        place = candidates[0]
        timezone = place.get("timezone")
        if not timezone:
            raise WeatherServiceError("invalid_location", "地点数据缺少时区，无法安全标注当地时间。")
        params: dict[str, Any] = {
            "latitude": place["latitude"], "longitude": place["longitude"],
            "timezone": timezone, "temperature_unit": "celsius", "wind_speed_unit": "kmh",
        }
        if mode == "current":
            params["current"] = "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
        else:
            params["daily"] = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            params["forecast_days"] = forecast_days
        payload = self._get_json(self.FORECAST_URL, params)
        data = payload.get("current" if mode == "current" else "daily")
        if not isinstance(data, dict):
            raise WeatherServiceError("invalid_response", "天气服务返回了不完整的数据。")

        result = {
            "status": "succeeded", "mode": mode, "location": self._candidate(place),
            "data": data, "timezone": timezone, "source": "Open-Meteo",
            "attribution": self.ATTRIBUTION,
        }
        self._cache[cache_key] = (self._clock() + self._cache_ttl_s, copy.deepcopy(result))
        return result

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.get(url, params=params, timeout=self._timeout_s)
        except httpx.TimeoutException as exc:
            raise WeatherServiceError("timeout", "天气服务请求超时，请稍后重试。") from exc
        except httpx.HTTPError as exc:
            raise WeatherServiceError("network_error", "无法连接天气服务，请检查网络后重试。") from exc
        if response.status_code == 429:
            raise WeatherServiceError("rate_limited", "天气服务暂时限流，请稍后重试。")
        if response.status_code >= 400:
            raise WeatherServiceError("upstream_error", f"天气服务返回 HTTP {response.status_code}。")
        try:
            payload = response.json()
        except ValueError as exc:
            raise WeatherServiceError("invalid_response", "天气服务返回了无效数据。") from exc
        if not isinstance(payload, dict):
            raise WeatherServiceError("invalid_response", "天气服务返回了无效数据。")
        if payload.get("error"):
            raise WeatherServiceError("upstream_error", "天气服务未能处理该地点请求。")
        return payload

    @staticmethod
    def _candidate(place: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": place.get("name", ""),
            "admin1": place.get("admin1", ""),
            "country": place.get("country", ""),
            "timezone": place.get("timezone", ""),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
        }
