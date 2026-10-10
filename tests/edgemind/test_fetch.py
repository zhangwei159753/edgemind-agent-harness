import socket

from edgemind.harness.integrations.fetch import SafePageFetcher


class FakeResponse:
    def __init__(self, status=200, headers=None, body=b""):
        self.status = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self._body = body

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, size=-1):
        return self._body if size < 0 else self._body[:size]


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.requested_path = None
        self.headers = {}

    def putrequest(self, method, path, **kwargs):
        self.requested_path = path

    def putheader(self, name, value):
        self.headers[name] = value

    def endheaders(self):
        pass

    def getresponse(self):
        return self.response

    def close(self):
        pass


def test_fetch_reads_allowed_html_and_removes_script_content():
    response = FakeResponse(body=b"<html><title>Example</title><script>ignore me</script><p>Useful text</p></html>")
    opened = []

    def factory(*args, **kwargs):
        conn = FakeConnection(response)
        opened.append((args, conn))
        return conn

    fetcher = SafePageFetcher(
        resolver=lambda host: ["93.184.216.34"], connection_factory=factory
    )
    result = fetcher.fetch("https://example.org/page", {"https://example.org/page"})

    assert result["status"] == "succeeded"
    assert "Useful text" in result["text"]
    assert "ignore me" not in result["text"]
    assert opened[0][1].headers["Host"] == "example.org"


def test_fetch_rejects_url_outside_current_result_allowlist():
    opened = []
    fetcher = SafePageFetcher(
        resolver=lambda host: ["93.184.216.34"],
        connection_factory=lambda *args, **kwargs: opened.append(args)
    )

    try:
        fetcher.fetch("https://not-in-results.example/page", {"https://example.org/page"})
    except Exception as error:
        assert getattr(error, "category", None) == "not_allowed"
    else:
        raise AssertionError("fetch must only read allowlisted URLs")

    assert opened == []


def test_fetch_rejects_localhost_and_private_addresses_before_connect():
    opened = []
    fetcher = SafePageFetcher(
        resolver=lambda host: ["127.0.0.1"],
        connection_factory=lambda *args, **kwargs: opened.append(args)
    )

    try:
        fetcher.fetch("http://localhost/private", {"http://localhost/private"})
    except Exception as error:
        assert getattr(error, "category", None) == "unsafe_url"
    else:
        raise AssertionError("localhost must be rejected")

    assert opened == []


def test_fetch_rechecks_redirect_target_before_opening_second_connection():
    responses = [FakeResponse(status=302, headers={"Location": "http://127.0.0.1/admin"})]
    opened = []

    def factory(*args, **kwargs):
        opened.append(args)
        return FakeConnection(responses.pop(0))

    fetcher = SafePageFetcher(
        resolver=lambda host: ["93.184.216.34"] if host == "example.org" else ["127.0.0.1"],
        connection_factory=factory,
    )
    try:
        fetcher.fetch("https://example.org/page", {"https://example.org/page"})
    except Exception as error:
        assert getattr(error, "category", None) == "unsafe_url"
    else:
        raise AssertionError("redirect to private address must be rejected")

    assert len(opened) == 1


def test_fetch_rejects_non_text_mime_and_over_limit_response():
    fetcher = SafePageFetcher(
        resolver=lambda host: ["93.184.216.34"],
        connection_factory=lambda *args, **kwargs: FakeConnection(
            FakeResponse(headers={"Content-Type": "application/octet-stream"}, body=b"data")
        ),
    )
    try:
        fetcher.fetch("https://example.org/file", {"https://example.org/file"})
    except Exception as error:
        assert getattr(error, "category", None) == "unsupported_mime"
    else:
        raise AssertionError("binary MIME must be rejected")

    oversized = SafePageFetcher(
        max_bytes=4, resolver=lambda host: ["93.184.216.34"],
        connection_factory=lambda *args, **kwargs: FakeConnection(
            FakeResponse(headers={"Content-Type": "text/plain"}, body=b"12345")
        ),
    )
    try:
        oversized.fetch("https://example.org/large", {"https://example.org/large"})
    except Exception as error:
        assert getattr(error, "category", None) == "too_large"
    else:
        raise AssertionError("body above configured cap must be rejected")


def test_fetch_rejects_non_http_scheme_and_credentials():
    opened = []
    fetcher = SafePageFetcher(
        resolver=lambda host: ["93.184.216.34"],
        connection_factory=lambda *args, **kwargs: opened.append(args)
    )
    for url in ("file:///etc/passwd", "https://user:pass@example.org/"):
        try:
            fetcher.fetch(url, {url})
        except Exception as error:
            assert getattr(error, "category", None) == "unsafe_url"
        else:
            raise AssertionError(f"unsafe URL accepted: {url}")

    assert opened == []


def test_fetch_maps_socket_timeout_to_safe_error():
    class TimeoutConnection(FakeConnection):
        def getresponse(self):
            raise socket.timeout("private upstream detail")

    def factory(*args, **kwargs):
        return TimeoutConnection(FakeResponse())

    fetcher = SafePageFetcher(
        resolver=lambda host: ["93.184.216.34"], connection_factory=factory
    )
    try:
        fetcher.fetch("https://example.org/page", {"https://example.org/page"})
    except Exception as error:
        assert getattr(error, "category", None) == "timeout"
        assert "private upstream detail" not in str(error)
    else:
        raise AssertionError("socket timeout must be mapped")
