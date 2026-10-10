import httpx

from edgemind.harness.integrations.search import BraveSearchClient


def test_search_normalizes_results_and_keeps_source_metadata():
    seen = {}

    def respond(request):
        seen["key"] = request.headers.get("x-subscription-token")
        seen["query"] = request.url.params.get("q")
        return httpx.Response(200, json={"web": {"results": [{
            "title": "Agent memory overview",
            "url": "https://example.org/agent-memory",
            "description": "A source summary.",
            "page_age": "2026-10-09T10:00:00"
        }]}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = BraveSearchClient(api_key="test-secret", client=client).search("agent memory")

    assert seen == {"key": "test-secret", "query": "agent memory"}
    assert result["status"] == "succeeded"
    assert result["results"] == [{
        "title": "Agent memory overview",
        "url": "https://example.org/agent-memory",
        "snippet": "A source summary.",
        "published_at": "2026-10-09T10:00:00",
        "source": "example.org"
    }]
    assert "test-secret" not in str(result)


def test_search_without_api_key_fails_with_safe_configuration_category():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("must not send request"))
    ))
    try:
        BraveSearchClient(api_key="", client=client).search("weather")
    except Exception as error:
        assert getattr(error, "category", None) == "not_configured"
        assert "secret" not in str(error).lower()
    else:
        raise AssertionError("missing key must not search")
    finally:
        client.close()


def test_search_rejects_query_over_brave_limit_before_request():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("invalid query must not send request"))
    ))
    try:
        BraveSearchClient(api_key="test-secret", client=client).search("x" * 601)
    except ValueError as error:
        assert "600" in str(error)
    else:
        raise AssertionError("query longer than 600 chars must be rejected")
    finally:
        client.close()


def test_search_rejects_count_above_five_before_request():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("invalid count must not send request"))
    ))
    try:
        BraveSearchClient(api_key="test-secret", client=client).search("weather", count=6)
    except ValueError as error:
        assert "5" in str(error)
    else:
        raise AssertionError("count above five must be rejected")
    finally:
        client.close()


def test_search_maps_rate_limit_and_never_exposes_upstream_body():
    def respond(request):
        assert request.headers["x-subscription-token"] == "test-secret"
        return httpx.Response(429, text="private upstream diagnostic")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        try:
            BraveSearchClient(api_key="test-secret", client=client).search("weather")
        except Exception as error:
            assert getattr(error, "category", None) == "rate_limited"
            assert "private upstream diagnostic" not in str(error)
            assert "test-secret" not in str(error)
        else:
            raise AssertionError("HTTP 429 must be reported")
