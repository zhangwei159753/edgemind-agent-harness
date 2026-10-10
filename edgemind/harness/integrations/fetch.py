from __future__ import annotations

import html
import http.client
import ipaddress
import socket
import ssl
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit
from typing import Callable


class FetchServiceError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class _VisibleText(HTMLParser):
    BLOCKED = {"script", "style", "noscript", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._blocked = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.BLOCKED:
            self._blocked += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "section", "article"} and not self._blocked:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.BLOCKED and self._blocked:
            self._blocked -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "li", "h1", "h2", "h3", "section", "article"} and not self._blocked:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._blocked:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.parts.append(data)


class SafePageFetcher:
    REDIRECTS = {301, 302, 303, 307, 308}

    def __init__(self, timeout_s: float = 10.0, max_bytes: int = 2 * 1024 * 1024,
                 resolver: Callable[[str], list[str]] | None = None, connection_factory=None,
                 max_text_chars: int = 12000):
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.max_text_chars = max_text_chars
        self._resolver = resolver or self._resolve
        self._connection_factory = connection_factory or self._connection

    def fetch(self, url: str, allowed_urls: set[str]) -> dict:
        if not isinstance(url, str) or self._canonical(url) not in {
            self._canonical(item) for item in (allowed_urls or set())
        }:
            raise FetchServiceError("not_allowed", "只能读取本轮搜索结果或你刚提供的网页链接。")
        current_url = url
        for redirect_count in range(4):
            parsed, address, port = self._validate_and_resolve(current_url)
            connection = self._connection_factory(parsed.scheme, address, port, parsed.hostname, self.timeout_s)
            try:
                path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                connection.putrequest("GET", path, skip_accept_encoding=True)
                host_header = parsed.hostname
                if parsed.port is not None:
                    host_header = f"{host_header}:{parsed.port}"
                connection.putheader("Host", host_header)
                connection.putheader("User-Agent", "EdgeMind/1.0")
                connection.putheader("Accept", "text/html, text/plain;q=0.9")
                connection.putheader("Connection", "close")
                connection.endheaders()
                response = connection.getresponse()
                if response.status in self.REDIRECTS:
                    if redirect_count >= 3:
                        raise FetchServiceError("redirect_limit", "网页跳转次数超过上限。")
                    location = response.getheader("Location")
                    if not location:
                        raise FetchServiceError("upstream_error", "网页返回了无目标地址的跳转。")
                    current_url = urljoin(current_url, location)
                    continue
                if response.status >= 400:
                    raise FetchServiceError("upstream_error", f"网页服务器返回 HTTP {response.status}。")
                content_type = (response.getheader("Content-Type", "").split(";", 1)[0].strip().lower())
                if content_type not in {"text/html", "text/plain"}:
                    raise FetchServiceError("unsupported_mime", "网页读取只支持 HTML 和纯文本。")
                length = response.getheader("Content-Length")
                if length:
                    try:
                        if int(length) > self.max_bytes:
                            raise FetchServiceError("too_large", "网页正文超过 2 MiB 上限。")
                    except ValueError:
                        pass
                body = response.read(self.max_bytes + 1)
                if len(body) > self.max_bytes:
                    raise FetchServiceError("too_large", "网页正文超过 2 MiB 上限。")
                charset = "utf-8"
                if ";" in response.getheader("Content-Type", ""):
                    for part in response.getheader("Content-Type").split(";")[1:]:
                        name, _, value = part.strip().partition("=")
                        if name.lower() == "charset" and value:
                            charset = value.strip('"\'')
                text = body.decode(charset, errors="replace")
                title = ""
                if content_type == "text/html":
                    parser = _VisibleText()
                    parser.feed(text)
                    title = " ".join(" ".join(parser.title_parts).split())[:300]
                    text = " ".join(" ".join(parser.parts).split())
                else:
                    text = " ".join(text.split())
                final = self._canonical(current_url)
                return {"status": "succeeded", "url": url, "final_url": final,
                        "title": title, "text": text[:self.max_text_chars],
                        "source": parsed.hostname, "source_type": content_type}
            except (socket.timeout, TimeoutError) as exc:
                raise FetchServiceError("timeout", "网页读取超时，请稍后重试。") from exc
            except OSError as exc:
                raise FetchServiceError("network_error", "无法连接目标网页，请稍后重试。") from exc
            finally:
                connection.close()
        raise FetchServiceError("redirect_limit", "网页跳转次数超过上限。")

    def _validate_and_resolve(self, url: str):
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise FetchServiceError("unsafe_url", "网页链接格式无效。") from exc
        if (parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or
                parsed.username is not None or parsed.password is not None or
                parsed.hostname.lower() == "localhost" or parsed.hostname.lower().endswith(".local")):
            raise FetchServiceError("unsafe_url", "网页读取仅允许公开 HTTP(S) 地址。")
        expected_port = 443 if parsed.scheme.lower() == "https" else 80
        if port not in (None, expected_port):
            raise FetchServiceError("unsafe_url", "网页读取只允许 HTTP 80 或 HTTPS 443 端口。")
        try:
            addresses = self._resolver(parsed.hostname)
        except (OSError, socket.gaierror) as exc:
            raise FetchServiceError("dns_error", "无法解析目标网页地址。") from exc
        if not addresses:
            raise FetchServiceError("dns_error", "目标网页没有可用地址。")
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError as exc:
                raise FetchServiceError("unsafe_url", "目标网页解析到了无效 IP 地址。") from exc
            if not ip.is_global:
                raise FetchServiceError("unsafe_url", "网页读取已阻止本机、私有或保留 IP 地址。")
        return parsed, addresses[0], expected_port

    @staticmethod
    def _resolve(host: str) -> list[str]:
        return list(dict.fromkeys(
            entry[4][0] for entry in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        ))

    @staticmethod
    def _connection(scheme: str, address: str, port: int, hostname: str, timeout: float):
        if scheme.lower() == "https":
            return http.client.HTTPSConnection(
                address, port, timeout=timeout, context=ssl.create_default_context(),
                server_hostname=hostname,
            )
        return http.client.HTTPConnection(address, port, timeout=timeout)

    @staticmethod
    def _canonical(url: str) -> str:
        try:
            parsed = urlsplit(url)
            if not parsed.hostname:
                return ""
            scheme = parsed.scheme.lower()
            host = parsed.hostname.encode("idna").decode("ascii").lower()
            port = parsed.port
            default_port = 443 if scheme == "https" else 80
            netloc = host if port in (None, default_port) else f"{host}:{port}"
            return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))
        except (ValueError, UnicodeError):
            return ""
