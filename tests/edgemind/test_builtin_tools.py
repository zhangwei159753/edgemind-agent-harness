from edgemind.harness.builtin_tools import BuiltinToolRegistry


class WeatherStub:
    def get_weather(self, location, mode="current", forecast_days=1):
        return {"status": "succeeded", "location": {"name": location},
                "mode": mode, "forecast_days": forecast_days}


class SearchStub:
    def search(self, query, count=5, country="CN", search_lang="zh-hans", freshness=None):
        return {"status": "succeeded", "query": query,
                "results": [{"url": "https://example.org/a", "title": "A"}],
                "result_count": 1}


class FetchStub:
    def __init__(self):
        self.calls = []

    def fetch(self, url, allowed_urls):
        self.calls.append((url, set(allowed_urls)))
        return {"status": "succeeded", "final_url": url, "text": "page"}


def test_registry_exposes_weather_when_brave_key_is_missing(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    registry = BuiltinToolRegistry(weather=WeatherStub(), search=SearchStub(), fetcher=FetchStub())

    names = [item["function"]["name"] for item in registry.model_tools()]

    assert names == ["edgemind__get_weather"]
    statuses = {item["id"]: item for item in registry.status()}
    assert statuses["open_meteo"]["ready"] is True
    assert statuses["brave_search"]["ready"] is False


def test_registry_exposes_all_tools_when_brave_key_exists_without_returning_key(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "never-return-this-key")
    registry = BuiltinToolRegistry(weather=WeatherStub(), search=SearchStub(), fetcher=FetchStub())

    names = [item["function"]["name"] for item in registry.model_tools()]
    status = registry.status()

    assert names == [
        "edgemind__get_weather", "edgemind__web_search", "edgemind__fetch_search_result"
    ]
    assert "never-return-this-key" not in str(status)
    assert {item["id"] for item in status} == {"open_meteo", "brave_search"}


def test_registry_validates_weather_args_before_dispatch(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    weather = WeatherStub()
    registry = BuiltinToolRegistry(weather=weather, search=SearchStub(), fetcher=FetchStub())

    try:
        registry.execute("edgemind__get_weather", {
            "location": "Paris", "mode": "forecast", "forecast_days": 8
        })
    except ValueError as error:
        assert "1 and 7" in str(error)
    else:
        raise AssertionError("forecast outside the schema limit must be rejected")


def test_registry_dispatches_search_and_restricts_fetch_to_supplied_allowlist(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    fetcher = FetchStub()
    registry = BuiltinToolRegistry(weather=WeatherStub(), search=SearchStub(), fetcher=fetcher)

    search_result = registry.execute("edgemind__web_search", {"query": "agent"})
    allowed = {item["url"] for item in search_result["results"]}
    page = registry.execute("edgemind__fetch_search_result", {
        "url": "https://example.org/a"
    }, allowed_urls=allowed)

    assert search_result["result_count"] == 1
    assert page["text"] == "page"
    assert fetcher.calls == [("https://example.org/a", {"https://example.org/a"})]


def test_registry_rejects_unknown_tools_and_non_object_arguments(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    registry = BuiltinToolRegistry(weather=WeatherStub(), search=SearchStub(), fetcher=FetchStub())

    for name, arguments in (("edgemind__shell", {}), ("edgemind__get_weather", [])):
        try:
            registry.execute(name, arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid call accepted: {name}")
