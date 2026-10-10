import httpx

from edgemind.harness.integrations.weather import OpenMeteoWeatherClient


def test_get_weather_returns_current_conditions_with_timezone_and_attribution():
    def respond(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": [{
                "name": "杭州", "latitude": 30.25, "longitude": 120.17,
                "timezone": "Asia/Shanghai", "country": "中国", "admin1": "浙江"
            }]})
        assert request.url.host == "api.open-meteo.com"
        assert request.url.params["timezone"] == "Asia/Shanghai"
        return httpx.Response(200, json={"current": {
            "time": "2026-10-10T14:00", "temperature_2m": 24.1,
            "weather_code": 1, "precipitation": 0.0, "wind_speed_10m": 7.2
        }})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = OpenMeteoWeatherClient(client=client).get_weather("杭州")

    assert result["status"] == "succeeded"
    assert result["location"]["name"] == "杭州"
    assert result["location"]["timezone"] == "Asia/Shanghai"
    assert result["data"]["time"] == "2026-10-10T14:00"
    assert result["data"]["temperature_2m"] == 24.1
    assert "Open-Meteo" in result["attribution"]


def test_get_weather_requests_requested_forecast_days():
    def respond(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": [{
                "name": "Paris", "latitude": 48.86, "longitude": 2.35,
                "timezone": "Europe/Paris", "country": "France"
            }]})
        assert request.url.params["forecast_days"] == "3"
        return httpx.Response(200, json={"daily": {
            "time": ["2026-10-10", "2026-10-11", "2026-10-12"],
            "temperature_2m_max": [20, 21, 19],
            "temperature_2m_min": [12, 13, 11],
            "weather_code": [1, 2, 3]
        }})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = OpenMeteoWeatherClient(client=client).get_weather(
            "Paris, France", mode="forecast", forecast_days=3
        )

    assert result["status"] == "succeeded"
    assert result["mode"] == "forecast"
    assert len(result["data"]["time"]) == 3


def test_get_weather_requires_clarification_for_ambiguous_location():
    def respond(request):
        return httpx.Response(200, json={"results": [
            {"name": "Springfield", "latitude": 39.8, "longitude": -89.6,
             "timezone": "America/Chicago", "country": "United States", "admin1": "Illinois"},
            {"name": "Springfield", "latitude": 42.1, "longitude": -72.6,
             "timezone": "America/New_York", "country": "United States", "admin1": "Massachusetts"}
        ]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = OpenMeteoWeatherClient(client=client).get_weather("Springfield")

    assert result["status"] == "needs_clarification"
    assert [item["admin1"] for item in result["location_candidates"]] == [
        "Illinois", "Massachusetts"
    ]


def test_get_weather_rejects_invalid_forecast_days_before_network_request():
    def unexpected_request(request):
        raise AssertionError("invalid arguments must be rejected before HTTP")

    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client:
        try:
            OpenMeteoWeatherClient(client=client).get_weather(
                "Paris", mode="forecast", forecast_days=8
            )
        except ValueError as error:
            assert "1 and 7" in str(error)
        else:
            raise AssertionError("forecast_days=8 must be rejected")



def test_get_weather_reuses_short_ttl_cache_for_same_query():
    calls = []

    def respond(request):
        calls.append(request.url.host)
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": [{
                "name": "Oslo", "latitude": 59.91, "longitude": 10.75,
                "timezone": "Europe/Oslo", "country": "Norway"
            }]})
        return httpx.Response(200, json={"current": {
            "time": "2026-10-10T08:00", "temperature_2m": 9,
            "weather_code": 3, "precipitation": 0, "wind_speed_10m": 4
        }})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        weather = OpenMeteoWeatherClient(client=client)
        first = weather.get_weather("Oslo")
        second = weather.get_weather("Oslo")

    assert first == second
    assert calls == ["geocoding-api.open-meteo.com", "api.open-meteo.com"]


def test_get_weather_reports_rate_limit_without_leaking_response_body():
    def respond(request):
        return httpx.Response(429, text="private upstream detail")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        try:
            OpenMeteoWeatherClient(client=client).get_weather("Oslo")
        except Exception as error:
            assert getattr(error, "category", None) == "rate_limited"
            assert "private upstream detail" not in str(error)
        else:
            raise AssertionError("HTTP 429 must be reported")
