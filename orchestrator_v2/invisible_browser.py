"""Bounded read-only browser evidence for the OpenCode v2 plugin."""
from __future__ import annotations

import base64
import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import socket
import ssl
import secrets
import stat
import sys
import threading
import tempfile
from pathlib import Path
from http import HTTPStatus
from urllib.parse import urlsplit, urlunsplit


MAX_BODY_CHARS = 1200
MAX_SELECTOR_CHARS = 200
MAX_SELECTOR_TEXT_CHARS = 600
MAX_TITLE_CHARS = 240
MAX_SCREENSHOT_BYTES = 1_500_000
MAX_REQUEST_BYTES = 8192
MAX_PROXY_CONFIG_BYTES = 64 * 1024
MAX_PROXY_HEADER_BYTES = 32 * 1024
MAX_PROXY_RESOURCE_BYTES = 16 * 1024 * 1024
DOWNLOAD_MIME_TYPES = {
    "application/octet-stream",
    "application/x-download",
    "application/x-msdownload",
    "application/x-executable",
    "application/x-sh",
}
CGNAT = ipaddress.ip_network("100.64.0.0/10")
METADATA_IPS = {ipaddress.ip_address("168.63.129.16")}
METADATA_HOSTS = {
    "metadata.google",
    "metadata.google.internal",
    "metadata.azure.internal",
    "instance-data.ec2.internal",
    "metadata.tencentyun.com",
    "metadata.oraclecloud.com",
}
LOCAL_SUFFIXES = ("localhost", ".localhost", ".local", ".lan", ".internal")
READ_ONLY_METHODS = {"GET", "HEAD", "OPTIONS"}
FIREFOX_NETWORK_PREFS = {
    "media.peerconnection.enabled": False,
    "network.dns.disablePrefetch": True,
    "network.prefetch-next": False,
    "network.proxy.allow_bypass": False,
    # Firefox bypasses its configured proxy for loopback by default. Setting
    # this makes even localhost targets go through the loopback filter, where
    # they are rejected instead of reaching Cortana services directly.
    "network.proxy.allow_hijacking_localhost": True,
    "zoom.stealth.dns.no_local_resolution": True,
    "dom.serviceWorkers.enabled": False,
    "browser.download.useDownloadDir": False,
    "browser.helperApps.alwaysAsk.force": True,
}
BODY_TEXT_EXPRESSION = f"document.body ? document.body.innerText.slice(0, {MAX_BODY_CHARS + 1}) : ''"
TITLE_TEXT_EXPRESSION = f"document.title.slice(0, {MAX_TITLE_CHARS + 1})"
SELECTOR_TEXT_EXPRESSION = f"element => (element.textContent || '').slice(0, {MAX_SELECTOR_TEXT_CHARS + 1})"


class BrowserPolicyError(Exception):
    """An error whose public message is deliberately safe to return."""

    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code
        self.reason = reason


def _global_address(value: str) -> bool:
    address = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv4Address) and address in CGNAT:
        return False
    return (
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
    )


def _preferred_address(addresses: set[str]) -> str:
    """Prefer IPv4 for networks without usable IPv6, otherwise keep IPv6 support."""
    return min(addresses, key=lambda value: (ipaddress.ip_address(value).version != 4, int(ipaddress.ip_address(value))))


def _proxy_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _resolve_public(hostname: str, port: int) -> set[str]:
    normalized = hostname.rstrip(".").lower()
    if not normalized or normalized in METADATA_HOSTS or normalized == "localhost" or normalized.endswith(LOCAL_SUFFIXES):
        raise BrowserPolicyError("target_rejected", "browser target rejected by network policy")
    try:
        literal = ipaddress.ip_address(normalized)
    except ValueError:
        try:
            records = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except (OSError, UnicodeError, ValueError) as exc:
            raise BrowserPolicyError("target_rejected", "browser target could not be safely resolved") from exc
        addresses = {record[4][0].split("%", 1)[0] for record in records}
    else:
        addresses = {str(literal)}
    try:
        safe_addresses = bool(addresses) and all(_global_address(address) for address in addresses)
    except ValueError as exc:
        raise BrowserPolicyError("target_rejected", "browser target could not be safely resolved") from exc
    if not safe_addresses:
        raise BrowserPolicyError("target_rejected", "browser target rejected by network policy")
    if any(ipaddress.ip_address(address) in METADATA_IPS for address in addresses):
        raise BrowserPolicyError("target_rejected", "browser target rejected by network policy")
    return addresses


def _public_url(value: str, same_origin: str | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise BrowserPolicyError("invalid_request", "browser URL is invalid or over-sized")
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("scheme or host")
        # Reject even empty user-info such as https://@example.org/.
        if "@" in parsed.netloc or parsed.username is not None or parsed.password is not None:
            raise ValueError("userinfo")
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme.lower() == "https" else 80)
        expected_port = 443 if parsed.scheme.lower() == "https" else 80
        if port != expected_port:
            raise ValueError("port")
    except (ValueError, UnicodeError) as exc:
        raise BrowserPolicyError("target_rejected", "browser target rejected by network policy") from exc
    if same_origin is not None:
        try:
            parent = urlsplit(same_origin)
            parent_port = parent.port if parent.port is not None else (443 if parent.scheme.lower() == "https" else 80)
            origin = (parsed.scheme.lower(), parsed.hostname.lower(), port)
            parent_origin = (parent.scheme.lower(), (parent.hostname or "").lower(), parent_port)
        except ValueError as exc:
            raise BrowserPolicyError("invalid_request", "browser navigation is invalid") from exc
        if origin != parent_origin:
            raise BrowserPolicyError("target_rejected", "secondary navigation must stay on the original origin")
    _resolve_public(parsed.hostname, port)
    return value


def _trusted_proxy() -> dict[str, str]:
    config_name = os.environ.get("V2_INVISIBLE_BROWSER_PROXY_CONFIG", "").strip()
    if not config_name:
        raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy is required")
    path = Path(config_name).expanduser()
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_size > MAX_PROXY_CONFIG_BYTES
        ):
            raise ValueError("unsafe proxy file")
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, UnicodeError) as exc:
        raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy configuration is unavailable or unsafe") from exc

    proxy = config.get("proxy") if isinstance(config, dict) else None
    if not isinstance(proxy, dict):
        raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy configuration is invalid")
    host = proxy.get("host")
    username = proxy.get("user")
    password = proxy.get("pass")
    try:
        port = int(proxy.get("port"))
    except (TypeError, ValueError) as exc:
        raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy configuration is invalid") from exc
    if (
        not isinstance(host, str)
        or not host.strip()
        or not isinstance(username, str)
        or not username
        or not isinstance(password, str)
        or not password
        or not 1 <= port <= 65535
    ):
        raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy configuration is invalid")

    raw_host = host.strip()
    if "://" in raw_host:
        try:
            parsed = urlsplit(raw_host)
            valid = (
                parsed.scheme.lower() == "https"
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and parsed.path in {"", "/"}
                and not parsed.query
                and not parsed.fragment
                and (parsed.port is None or parsed.port == port)
            )
        except ValueError as exc:
            raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy configuration is invalid") from exc
        if not valid:
            raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy configuration is invalid")
        proxy_host = parsed.hostname or ""
    else:
        if any(character in raw_host for character in "/@?# \t\r\n"):
            raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy configuration is invalid")
        proxy_host = raw_host[1:-1] if raw_host.startswith("[") and raw_host.endswith("]") else raw_host
    try:
        proxy_addresses = _resolve_public(proxy_host, port)
    except BrowserPolicyError as exc:
        raise BrowserPolicyError("proxy_unavailable", "trusted browser proxy is not externally routable") from exc
    pinned_proxy_ip = _preferred_address(proxy_addresses)
    # Retain the hostname for TLS certificate validation; the connector and
    # raw CONNECT path separately pin its socket to this checked address.
    server_authority = f"[{proxy_host}]:{port}" if ":" in proxy_host else f"{proxy_host}:{port}"
    server = urlunsplit(("https", server_authority, "", "", ""))
    return {
        "server": server,
        "hostname": proxy_host,
        "pinned_ip": pinned_proxy_ip,
        "username": username,
        # The configured residential proxy expects this provider-supported
        # session suffix to keep one egress IP stable for the browser lifetime.
        "password": f"{password}_session-{secrets.token_hex(4)}_lifetime-10m",
    }


def _text(value: object, limit: int = MAX_BODY_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def _validate_filter_request(method: str, target: str) -> tuple[str, int, str]:
    method = method.upper()
    if method == "CONNECT":
        try:
            parsed = urlsplit("//" + target)
            hostname = parsed.hostname
            port = parsed.port if parsed.port is not None else 443
        except ValueError as exc:
            raise BrowserPolicyError("target_rejected", "tunnel rejected") from exc
        if (
            not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or port != 443
        ):
            raise BrowserPolicyError("target_rejected", "tunnel rejected")
        addresses = _resolve_public(hostname, port)
        return hostname, port, _preferred_address(addresses)
    if method not in READ_ONLY_METHODS:
        raise BrowserPolicyError("target_rejected", "non-read-only request")
    try:
        parsed = urlsplit(target)
        if parsed.scheme.lower() != "http" or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError("target")
        port = parsed.port if parsed.port is not None else 80
    except ValueError as exc:
        raise BrowserPolicyError("target_rejected", "request rejected") from exc
    if port != 80:
        raise BrowserPolicyError("target_rejected", "request rejected")
    addresses = _resolve_public(parsed.hostname, port)
    return parsed.hostname, port, _preferred_address(addresses)


def _is_websocket_upgrade(headers: list[tuple[str, str]]) -> bool:
    values = {name.lower(): value for name, value in headers}
    connection_tokens = {
        token.strip().lower()
        for name, value in headers
        if name.lower() == "connection"
        for token in value.split(",")
        if token.strip()
    }
    return "websocket" in values.get("upgrade", "").lower() or "upgrade" in connection_tokens


def _safe_failure(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, BrowserPolicyError):
        return {"status": "ERROR", "code": exc.code, "reason": exc.reason}
    if isinstance(exc, (TimeoutError,)):
        code, reason = "timeout", "browser investigation timed out"
    elif isinstance(exc, ImportError):
        code, reason = "runtime_unavailable", "browser runtime unavailable"
    elif isinstance(exc, (json.JSONDecodeError, UnicodeError)):
        code, reason = "invalid_request", "browser request is invalid"
    else:
        code, reason = "browser_failure", "browser investigation failed safely"
    return {"status": "ERROR", "code": code, "reason": reason}


class _LoopbackFilteringProxy:
    """A narrow loopback request filter which forwards only through upstream."""

    def __init__(self, upstream: dict[str, str]):
        parsed = urlsplit(upstream["server"])
        if parsed.scheme != "https" or not parsed.hostname or not parsed.port:
            raise BrowserPolicyError("proxy_unavailable", "trusted encrypted HTTP proxy configuration is required")
        self._upstream = upstream
        self._upstream_host = upstream.get("hostname", parsed.hostname)
        self._upstream_ip = upstream.get("pinned_ip", parsed.hostname)
        self._upstream_port = parsed.port
        self._local_username = f"v2-{secrets.token_hex(8)}"
        self._local_password = secrets.token_urlsafe(24)
        self._local_auth = "Basic " + base64.b64encode(
            f"{self._local_username}:{self._local_password}".encode("ascii")
        ).decode("ascii")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None
        self._session = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_failed = False
        self._writers: set[asyncio.StreamWriter] = set()
        self._upstream_writers: set[asyncio.StreamWriter] = set()
        self._handler_tasks: set[asyncio.Task] = set()
        self._port = 0
        self._requests = 0
        self._counter_lock = threading.Lock()

    @property
    def checked_requests(self) -> int:
        with self._counter_lock:
            return self._requests

    @property
    def browser_settings(self) -> dict[str, str]:
        return {
            "server": f"http://127.0.0.1:{self._port}",
            "username": self._local_username,
            "password": self._local_password,
        }

    def __enter__(self) -> "_LoopbackFilteringProxy":
        self._thread = threading.Thread(target=self._run, name="v2-browser-egress", daemon=True)
        self._thread.start()
        if not self._ready.wait(5) or self._startup_failed or not self._port:
            self.close()
            raise BrowserPolicyError("proxy_unavailable", "local browser network filter could not start")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._close_async(), loop).result(timeout=6)
            except Exception:
                pass
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_async())
        except Exception:
            self._startup_failed = True
            self._ready.set()
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _start_async(self) -> None:
        import aiohttp

        upstream_host = self._upstream_host.lower().rstrip(".")
        upstream_ip = self._upstream_ip

        class PinnedProxyResolver(aiohttp.abc.AbstractResolver):
            async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
                if host.lower().rstrip(".") != upstream_host:
                    raise OSError("unexpected resolver request")
                address = ipaddress.ip_address(upstream_ip)
                return [{
                    "hostname": host,
                    "host": upstream_ip,
                    "port": port,
                    "family": socket.AF_INET6 if address.version == 6 else socket.AF_INET,
                    "proto": 0,
                    "flags": 0,
                }]

            async def close(self) -> None:
                return None

        self._session = aiohttp.ClientSession(
            auto_decompress=False,
            connector=aiohttp.TCPConnector(
                limit=16,
                force_close=True,
                resolver=PinnedProxyResolver(),
                ssl=_proxy_ssl_context(),
            ),
            timeout=aiohttp.ClientTimeout(total=30, connect=10, sock_read=20),
            trust_env=False,
        )
        self._server = await asyncio.start_server(
            self._handle_client,
            "127.0.0.1",
            0,
            limit=MAX_PROXY_HEADER_BYTES,
        )
        self._port = int(self._server.sockets[0].getsockname()[1])

    async def _close_async(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        upstreams = list(self._upstream_writers)
        for writer in upstreams:
            writer.transport.abort()
        handlers = [task for task in self._handler_tasks if task is not asyncio.current_task()]
        for task in handlers:
            task.cancel()
        if handlers:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.gather(*handlers, return_exceptions=True), timeout=2)
        writers = list(self._writers)
        for writer in writers:
            writer.close()
        if writers:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*(writer.wait_closed() for writer in writers), return_exceptions=True),
                    timeout=1,
                )
        if self._session is not None:
            await self._session.close()
            # Let TLS transports finish close-notify before this loop stops.
            await asyncio.sleep(0.25)

    async def _read_request(self, reader: asyncio.StreamReader) -> tuple[str, str, list[tuple[str, str]]]:
        first = await asyncio.wait_for(reader.readline(), timeout=8)
        if not first or len(first) > 8192:
            raise ValueError("request line")
        parts = first.decode("latin-1").strip().split()
        if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
            raise ValueError("request line")
        headers: list[tuple[str, str]] = []
        total = len(first)
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=8)
            total += len(line)
            if total > MAX_PROXY_HEADER_BYTES or not line:
                raise ValueError("headers")
            if line in (b"\r\n", b"\n"):
                break
            name, separator, value = line.partition(b":")
            if not separator or not name:
                raise ValueError("header")
            headers.append((name.decode("latin-1").strip(), value.decode("latin-1").strip()))
        return parts[0].upper(), parts[1], headers

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        task = asyncio.current_task()
        if task is not None:
            self._handler_tasks.add(task)
        try:
            method, target, headers = await self._read_request(reader)
            local_auth = [value for name, value in headers if name.lower() == "proxy-authorization"]
            if len(local_auth) != 1 or not hmac.compare_digest(local_auth[0], self._local_auth):
                await self._reply(writer, 407, 'Proxy-Authenticate: Basic realm="v2-browser"')
                return
            if method == "CONNECT":
                await self._handle_connect(reader, writer, target)
                return
            await self._handle_http(writer, method, target, headers)
        except BrowserPolicyError:
            await self._reply(writer, 403)
        except Exception:
            await self._reply(writer, 502)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            self._writers.discard(writer)
            if task is not None:
                self._handler_tasks.discard(task)

    async def _reply(self, writer: asyncio.StreamWriter, status: int, extra_header: str = "") -> None:
        phrase = HTTPStatus(status).phrase
        extra = f"{extra_header}\r\n" if extra_header else ""
        payload = f"HTTP/1.1 {status} {phrase}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n".encode("ascii")
        writer.write(payload)
        with contextlib.suppress(Exception):
            await writer.drain()

    def _count_checked(self) -> None:
        with self._counter_lock:
            self._requests += 1

    @staticmethod
    def _connection_tokens(headers: list[tuple[str, str]]) -> set[str]:
        tokens: set[str] = set()
        for name, value in headers:
            if name.lower() == "connection":
                tokens.update(part.strip().lower() for part in value.split(",") if part.strip())
        return tokens

    async def _handle_http(
        self,
        writer: asyncio.StreamWriter,
        method: str,
        target: str,
        headers: list[tuple[str, str]],
    ) -> None:
        import aiohttp

        self._count_checked()
        if method not in READ_ONLY_METHODS:
            await self._reply(writer, 405)
            return
        try:
            hostname, port, pinned_ip = _validate_filter_request(method, target)
        except BrowserPolicyError:
            await self._reply(writer, 403)
            return
        if _is_websocket_upgrade(headers):
            await self._reply(writer, 403)
            return
        try:
            content_lengths = [
                int(value or "0")
                for name, value in headers
                if name.lower() == "content-length"
            ]
            if any(length != 0 for length in content_lengths) or any(
                name.lower() == "transfer-encoding" for name, _value in headers
            ):
                await self._reply(writer, 405)
                return
        except ValueError:
            await self._reply(writer, 400)
            return

        hop_by_hop = {
            "connection", "host", "keep-alive", "proxy-authorization", "proxy-connection",
            "te", "trailer", "transfer-encoding", "upgrade",
        } | self._connection_tokens(headers)
        forwarded = [(name, value) for name, value in headers if name.lower() not in hop_by_hop]
        parsed_target = urlsplit(target)
        ip_address = ipaddress.ip_address(pinned_ip)
        ip_authority = f"[{pinned_ip}]" if isinstance(ip_address, ipaddress.IPv6Address) else pinned_ip
        pinned_target = urlunsplit(("http", ip_authority, parsed_target.path or "/", parsed_target.query, ""))
        original_host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed_target.port is not None:
            original_host += f":{parsed_target.port}"
        forwarded.append(("Host", original_host))
        forwarded.append(("Connection", "close"))
        response_started = False
        try:
            async with self._session.request(
                method,
                pinned_target,
                headers=forwarded,
                allow_redirects=False,
                proxy=self._upstream["server"],
                proxy_auth=aiohttp.BasicAuth(self._upstream["username"], self._upstream["password"]),
            ) as response:
                response_headers = {name.lower(): value for name, value in response.headers.items()}
                disposition = response_headers.get("content-disposition", "").lower()
                media_type = response_headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if "attachment" in disposition or media_type in DOWNLOAD_MIME_TYPES:
                    await self._reply(writer, 403)
                    return
                length = response.content_length
                body_allowed = method != "HEAD" and response.status not in {204, 304}
                if body_allowed and length is not None and length > MAX_PROXY_RESOURCE_BYTES:
                    await self._reply(writer, 413)
                    return
                phrase = HTTPStatus(response.status).phrase if response.status in HTTPStatus._value2member_map_ else "Response"
                writer.write(f"HTTP/1.1 {response.status} {phrase}\r\n".encode("ascii"))
                response_started = True
                response_hop = {
                    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                    "te", "trailer", "transfer-encoding", "upgrade",
                }
                for raw_name, raw_value in response.raw_headers:
                    name = raw_name.decode("latin-1")
                    if name.lower() not in response_hop:
                        writer.write(raw_name + b": " + raw_value + b"\r\n")
                if body_allowed and length is None:
                    writer.write(b"Transfer-Encoding: chunked\r\n")
                writer.write(b"Connection: close\r\n\r\n")
                await writer.drain()
                if body_allowed:
                    sent = 0
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        sent += len(chunk)
                        if sent > MAX_PROXY_RESOURCE_BYTES:
                            writer.close()
                            return
                        if length is None:
                            writer.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                        else:
                            writer.write(chunk)
                        await writer.drain()
                    if length is None:
                        writer.write(b"0\r\n\r\n")
                        await writer.drain()
        except Exception:
            if response_started:
                writer.close()
            else:
                await self._reply(writer, 502)

    async def _handle_connect(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter, target: str) -> None:
        self._count_checked()
        try:
            hostname, port, pinned_ip = _validate_filter_request("CONNECT", target)
        except BrowserPolicyError:
            await self._reply(client_writer, 403)
            return
        upstream_reader = None
        upstream_writer = None
        tunnel_started = False
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self._upstream_ip,
                    self._upstream_port,
                    ssl=_proxy_ssl_context(),
                    server_hostname=self._upstream_host,
                ),
                timeout=10,
            )
            self._upstream_writers.add(upstream_writer)
            credential = base64.b64encode(
                f"{self._upstream['username']}:{self._upstream['password']}".encode("utf-8")
            ).decode("ascii")
            target_ip = ipaddress.ip_address(pinned_ip)
            connect_host = f"[{pinned_ip}]" if isinstance(target_ip, ipaddress.IPv6Address) else pinned_ip
            upstream_writer.write((
                f"CONNECT {connect_host}:{port} HTTP/1.1\r\n"
                f"Host: {connect_host}:{port}\r\n"
                f"Proxy-Authorization: Basic {credential}\r\n"
                "Proxy-Connection: Keep-Alive\r\n\r\n"
            ).encode("ascii"))
            await upstream_writer.drain()
            status_line = await asyncio.wait_for(upstream_reader.readline(), timeout=10)
            if len(status_line) > 8192:
                raise OSError("proxy response")
            status_parts = status_line.decode("latin-1").split()
            if len(status_parts) < 2 or status_parts[1] != "200":
                await self._reply(client_writer, 502)
                return
            total = len(status_line)
            while True:
                line = await asyncio.wait_for(upstream_reader.readline(), timeout=10)
                total += len(line)
                if total > MAX_PROXY_HEADER_BYTES or not line:
                    raise OSError("proxy headers")
                if line in (b"\r\n", b"\n"):
                    break
            client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await client_writer.drain()
            tunnel_started = True
            await self._tunnel(client_reader, client_writer, upstream_reader, upstream_writer)
        except Exception:
            if not tunnel_started:
                with contextlib.suppress(Exception):
                    await self._reply(client_writer, 502)
        finally:
            if upstream_writer is not None:
                upstream_writer.close()
                try:
                    await asyncio.wait_for(upstream_writer.wait_closed(), timeout=1)
                except asyncio.CancelledError:
                    upstream_writer.transport.abort()
                    raise
                except Exception:
                    upstream_writer.transport.abort()
                finally:
                    self._upstream_writers.discard(upstream_writer)

    @staticmethod
    async def _tunnel(
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        upstream_reader: asyncio.StreamReader,
        upstream_writer: asyncio.StreamWriter,
    ) -> None:
        async def copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            while True:
                data = await asyncio.wait_for(reader.read(64 * 1024), timeout=30)
                if not data:
                    return
                writer.write(data)
                await writer.drain()

        tasks = {
            asyncio.create_task(copy(client_reader, upstream_writer)),
            asyncio.create_task(copy(upstream_reader, client_writer)),
        }
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def run_probe(request: dict[str, object]) -> dict[str, object]:
    proxy = _trusted_proxy()
    raw_url = request.get("url")
    if not isinstance(raw_url, str):
        raise BrowserPolicyError("invalid_request", "browser URL is invalid")
    url = _public_url(raw_url)
    navigate_to = request.get("navigate_to")
    if navigate_to is not None:
        if not isinstance(navigate_to, str):
            raise BrowserPolicyError("invalid_request", "secondary navigation is invalid")
        # Resolve and classify the requested URL now; enforce same-origin only
        # after the initial navigation's redirects establish the actual origin.
        navigate_to = _public_url(navigate_to)
    selector = request.get("selector")
    if selector is not None:
        if not isinstance(selector, str) or not selector or len(selector) > MAX_SELECTOR_CHARS or "javascript:" in selector.lower():
            raise BrowserPolicyError("invalid_request", "selector is invalid or over-sized")
    screenshot = request.get("screenshot", False)
    if not isinstance(screenshot, bool):
        raise BrowserPolicyError("invalid_request", "screenshot must be a boolean")

    from invisible_playwright import InvisiblePlaywright

    # The bundled Juggler ignores serviceWorkers/acceptDownloads in its context
    # dispatcher; the service-worker and download safety prefs are enforced by
    # Firefox itself, with response-type blocking in the filtering proxy too.
    extra_prefs = FIREFOX_NETWORK_PREFS
    with _LoopbackFilteringProxy(proxy) as filtered_proxy:
        with InvisiblePlaywright(headless=True, proxy=filtered_proxy.browser_settings, extra_prefs=extra_prefs) as browser:
            context = browser.new_context(service_workers="block", accept_downloads=False)
            try:
                page = context.new_page()
                page.goto(url, timeout=60_000, wait_until="domcontentloaded")
                _public_url(page.url)
                navigation_origin = page.url
                try:
                    page.wait_for_load_state("load", timeout=10_000)
                except Exception:
                    pass
                if navigate_to:
                    checked_navigation = _public_url(navigate_to, same_origin=navigation_origin)
                    page.goto(checked_navigation, timeout=60_000, wait_until="domcontentloaded")
                    _public_url(page.url, same_origin=navigation_origin)
                    try:
                        page.wait_for_load_state("load", timeout=10_000)
                    except Exception:
                        pass
                else:
                    _public_url(page.url)
                result: dict[str, object] = {
                    "status": "OK",
                    "url": page.url,
                    "title": _text(page.evaluate(TITLE_TEXT_EXPRESSION), MAX_TITLE_CHARS),
                    "ready_state": _text(page.evaluate("document.readyState"), 40),
                    "body_text": _text(page.evaluate(BODY_TEXT_EXPRESSION)),
                    "webdriver": page.evaluate("navigator.webdriver"),
                    "proxy_configured": True,
                }
                if selector:
                    result["selector_text"] = _text(
                        page.locator(selector).evaluate(SELECTOR_TEXT_EXPRESSION),
                        MAX_SELECTOR_TEXT_CHARS,
                    )
                if screenshot:
                    temp_dir = os.environ.get("V2_INVISIBLE_BROWSER_TMPDIR")
                    with tempfile.NamedTemporaryFile(suffix=".jpg", dir=temp_dir) as image:
                        page.screenshot(path=image.name, type="jpeg", quality=75, full_page=False)
                        image.flush()
                        image_size = os.fstat(image.fileno()).st_size
                        if image_size <= MAX_SCREENSHOT_BYTES:
                            image.seek(0)
                            data = image.read(MAX_SCREENSHOT_BYTES + 1)
                        else:
                            data = b""
                    if image_size <= MAX_SCREENSHOT_BYTES and len(data) == image_size:
                        result["screenshot_sha256"] = hashlib.sha256(data).hexdigest()
                        result["screenshot_bytes"] = image_size
                        result["screenshot_base64"] = base64.b64encode(data).decode("ascii")
                    else:
                        result["screenshot_omitted"] = "screenshot exceeded the bounded attachment size"
            finally:
                context.close()
        result["network_requests_checked"] = filtered_proxy.checked_requests
    return result


def main() -> int:
    try:
        raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise BrowserPolicyError("invalid_request", "browser request is over-sized")
        request = json.loads(raw.decode("utf-8"))
        if not isinstance(request, dict):
            raise BrowserPolicyError("invalid_request", "browser request must be an object")
        output = run_probe(request)
    except BaseException as exc:
        output = _safe_failure(exc)
    sys.stdout.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
