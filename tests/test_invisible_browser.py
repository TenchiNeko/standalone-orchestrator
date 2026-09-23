import asyncio
import base64
import io
import importlib.util
import json
import os
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator_v2 import invisible_browser as browser


GLOBAL = "93.184.216.34"
GLOBAL_V6 = "2606:4700:4700::1111"


def addr_info(*addresses):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)) for address in addresses]


class FakePage:
    def __init__(self, context):
        self.context = context
        self.url = "about:blank"
        self.screenshot_bytes = b"jpeg-image"

    def goto(self, url, **_kwargs):
        self.url = url

    def wait_for_load_state(self, *_args, **_kwargs):
        return None

    def title(self):
        return 'Example "quoted"'

    def evaluate(self, expression):
        return {
            "document.readyState": "complete",
            browser.BODY_TEXT_EXPRESSION: 'hello "world"\nline 😀',
            browser.TITLE_TEXT_EXPRESSION: 'Example "quoted"',
            "navigator.webdriver": False,
        }[expression]

    def screenshot(self, path, **_kwargs):
        with open(path, "wb") as handle:
            handle.write(self.screenshot_bytes)

    def locator(self, _selector):
        return FakeLocator()

    def text_content(self):
        return 'selector "text"\n😀'


class FakeLocator:
    def evaluate(self, _expression):
        return 'selector "text"\n😀'


class FakeContext:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def new_page(self):
        return FakePage(self)

    def close(self):
        self.closed = True


class FakeBrowser:
    last_context = None
    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeBrowser.last_kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def new_context(self, **kwargs):
        FakeBrowser.last_context = FakeContext(**kwargs)
        return FakeBrowser.last_context


class FakeFilteringProxy:
    def __init__(self, upstream):
        self.upstream = upstream
        self.checked_requests = 7
        self.browser_settings = {"server": "http://127.0.0.1:43123"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class InvisibleBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.proxy_config = Path(self.temp.name) / "proxy.json"
        self.proxy_config.write_text(json.dumps({"proxy": {
            "host": "proxy.example", "port": 3128, "user": "worker-user", "pass": "fake-proxy-secret",
        }}))
        self.proxy_config.chmod(0o600)
        self.env = patch.dict(os.environ, {
            "V2_INVISIBLE_BROWSER_PROXY_CONFIG": str(self.proxy_config),
            "V2_INVISIBLE_BROWSER_TMPDIR": self.temp.name,
        }, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dns = patch(
            "orchestrator_v2.invisible_browser.socket.getaddrinfo",
            side_effect=lambda host, *_args, **_kwargs: addr_info(GLOBAL_V6 if ":" in host else GLOBAL),
        )
        self.dns.start()
        self.addCleanup(self.dns.stop)

    def tearDown(self):
        self.temp.cleanup()

    def test_only_global_public_ipv4_ipv6_are_accepted(self):
        self.assertEqual(browser._public_url("https://93.184.216.34/"), "https://93.184.216.34/")
        self.assertEqual(browser._public_url(f"https://[{GLOBAL_V6}]/"), f"https://[{GLOBAL_V6}]/")

    def test_local_reserved_and_cgnat_addresses_are_rejected(self):
        rejected = (
            "127.0.0.1", "10.1.2.3", "172.16.0.1", "172.31.255.254",
            "192.168.1.1", "169.254.1.1", "100.64.0.1", "100.85.27.103",
            "0.0.0.0", "224.0.0.1", "192.0.2.1", "198.51.100.4",
            "::1", "fe80::1", "fc00::1", "fd12:3456::1", "2001:db8::1",
        )
        for address in rejected:
            with self.subTest(address=address):
                target = f"https://[{address}]/" if ":" in address else f"http://{address}/"
                with self.assertRaises(browser.BrowserPolicyError):
                    browser._public_url(target)

    def test_dns_answers_must_all_be_global_and_rechecked_for_rebinding(self):
        with patch("orchestrator_v2.invisible_browser.socket.getaddrinfo", return_value=addr_info(GLOBAL, "10.0.0.8")):
            with self.assertRaises(browser.BrowserPolicyError):
                browser._public_url("https://rebind.example/")
        with patch("orchestrator_v2.invisible_browser.socket.getaddrinfo", side_effect=[addr_info(GLOBAL), addr_info("127.0.0.1")]):
            self.assertEqual(browser._validate_filter_request("GET", "http://rebind.example/"), ("rebind.example", 80, GLOBAL))
            with self.assertRaises(browser.BrowserPolicyError):
                browser._validate_filter_request("GET", "http://rebind.example/next")

    def test_url_credentials_including_empty_userinfo_are_rejected(self):
        for value in ("https://user:pass@example.com/", "https://@example.com/", "https://user@example.com/"):
            with self.subTest(value=value), self.assertRaises(browser.BrowserPolicyError):
                browser._public_url(value)

    def test_same_origin_navigation_is_enforced(self):
        with self.assertRaises(browser.BrowserPolicyError):
            browser._public_url("https://other.example/", same_origin="https://example.com/")

    def test_secondary_navigation_origin_uses_the_actual_post_redirect_url(self):
        calls = []

        def redirect_page(page, target, **_kwargs):
            calls.append(target)
            page.url = "https://redirect.example/" if len(calls) == 1 else target

        module = types.SimpleNamespace(InvisiblePlaywright=FakeBrowser)
        with (
            patch.dict(sys.modules, {"invisible_playwright": module}),
            patch.object(browser, "_LoopbackFilteringProxy", FakeFilteringProxy),
            patch.object(FakePage, "goto", redirect_page),
            self.assertRaises(browser.BrowserPolicyError),
        ):
            browser.run_probe({"url": "https://example.com/", "navigate_to": "https://example.com/next"})
        self.assertEqual(calls, ["https://example.com/",])
        self.assertTrue(FakeBrowser.last_context.closed)

    def test_nonstandard_destination_ports_are_rejected(self):
        for value in ("http://example.com:8080/", "https://example.com:8443/"):
            with self.subTest(value=value), self.assertRaises(browser.BrowserPolicyError):
                browser._public_url(value)

    def test_proxy_policy_blocks_redirect_subresource_iframe_fetch_and_xhr_targets(self):
        request_kinds = (
            "redirect", "subresource", "iframe", "fetch", "xhr",
        )
        for kind in request_kinds:
            with self.subTest(kind=kind):
                with self.assertRaises(browser.BrowserPolicyError):
                    browser._validate_filter_request("GET", "http://192.168.1.10/private")
        for kind in ("redirect", "subresource", "iframe", "fetch", "xhr", "websocket"):
            with self.subTest(kind=kind):
                with self.assertRaises(browser.BrowserPolicyError):
                    browser._validate_filter_request("CONNECT", "100.85.27.103:443")

    def test_proxy_policy_rejects_mutating_methods_and_plain_websockets(self):
        with self.assertRaises(browser.BrowserPolicyError):
            browser._validate_filter_request("POST", "http://example.com/submit")
        self.assertTrue(browser._is_websocket_upgrade([("Connection", "keep-alive, Upgrade"), ("Upgrade", "websocket")]))
        self.assertFalse(browser._is_websocket_upgrade([("Connection", "keep-alive")]))
        proxy = browser._trusted_proxy()
        self.assertNotIn("bypass", proxy)

    def test_proxy_is_mandatory_owner_only_and_external(self):
        with patch.dict(os.environ, {"V2_INVISIBLE_BROWSER_PROXY_CONFIG": ""}):
            with self.assertRaises(browser.BrowserPolicyError) as err:
                browser._trusted_proxy()
        self.assertEqual(err.exception.code, "proxy_unavailable")
        self.assertEqual(browser._trusted_proxy()["server"], "https://proxy.example:3128")
        self.assertRegex(
            browser._trusted_proxy()["password"],
            r"^fake-proxy-secret_session-[0-9a-f]{8}_lifetime-10m$",
        )
        self.proxy_config.write_text(json.dumps({"proxy": {
            "host": "http://proxy.example", "port": 3128, "user": "worker-user", "pass": "fake-proxy-secret",
        }}))
        self.proxy_config.chmod(0o600)
        with self.assertRaises(browser.BrowserPolicyError):
            browser._trusted_proxy()
        self.proxy_config.chmod(0o644)
        with self.assertRaises(browser.BrowserPolicyError):
            browser._trusted_proxy()

    def test_proxy_local_addresses_fail_closed(self):
        with patch("orchestrator_v2.invisible_browser.socket.getaddrinfo", return_value=addr_info("100.85.27.103")):
            with self.assertRaises(browser.BrowserPolicyError):
                browser._trusted_proxy()

    def test_probe_uses_loopback_filter_blocks_service_workers_and_defaults_no_screenshot(self):
        module = types.SimpleNamespace(InvisiblePlaywright=FakeBrowser)
        with patch.dict(sys.modules, {"invisible_playwright": module}), patch.object(browser, "_LoopbackFilteringProxy", FakeFilteringProxy):
            result = browser.run_probe({"url": "https://example.com"})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["body_text"], 'hello "world" line 😀')
        self.assertEqual(result["title"], 'Example "quoted"')
        self.assertEqual(FakeBrowser.last_context.kwargs["service_workers"], "block")
        self.assertEqual(FakeBrowser.last_context.kwargs["accept_downloads"], False)
        self.assertNotIn("bypass", FakeBrowser.last_kwargs["proxy"])
        self.assertEqual(FakeBrowser.last_kwargs["proxy"]["server"], "http://127.0.0.1:43123")
        self.assertFalse(FakeBrowser.last_kwargs["extra_prefs"]["media.peerconnection.enabled"])
        self.assertFalse(FakeBrowser.last_kwargs["extra_prefs"]["network.proxy.allow_bypass"])
        self.assertTrue(FakeBrowser.last_kwargs["extra_prefs"]["network.proxy.allow_hijacking_localhost"])
        self.assertFalse(FakeBrowser.last_kwargs["extra_prefs"]["dom.serviceWorkers.enabled"])
        self.assertFalse(FakeBrowser.last_kwargs["extra_prefs"]["browser.download.useDownloadDir"])
        self.assertEqual(result["network_requests_checked"], 7)
        self.assertNotIn("screenshot_base64", result)
        self.assertTrue(FakeBrowser.last_context.closed)

    def test_dom_and_selector_text_are_bounded_before_model_output(self):
        observed = []

        def large_evaluate(_self, expression):
            observed.append(expression)
            if expression == browser.BODY_TEXT_EXPRESSION:
                return "b" * (browser.MAX_BODY_CHARS + 100)
            if expression == browser.TITLE_TEXT_EXPRESSION:
                return "t" * (browser.MAX_TITLE_CHARS + 100)
            return "complete" if expression == "document.readyState" else False

        def large_selector(_self, expression):
            observed.append(expression)
            return "s" * (browser.MAX_SELECTOR_TEXT_CHARS + 100)

        module = types.SimpleNamespace(InvisiblePlaywright=FakeBrowser)
        with patch.dict(sys.modules, {"invisible_playwright": module}), patch.object(browser, "_LoopbackFilteringProxy", FakeFilteringProxy), patch.object(FakePage, "evaluate", large_evaluate), patch.object(FakeLocator, "evaluate", large_selector):
            result = browser.run_probe({"url": "https://example.com", "selector": ".bounded"})
        self.assertEqual(len(result["body_text"]), browser.MAX_BODY_CHARS)
        self.assertEqual(len(result["title"]), browser.MAX_TITLE_CHARS)
        self.assertEqual(len(result["selector_text"]), browser.MAX_SELECTOR_TEXT_CHARS)
        self.assertIn(browser.BODY_TEXT_EXPRESSION, observed)
        self.assertIn(browser.TITLE_TEXT_EXPRESSION, observed)
        self.assertIn(browser.SELECTOR_TEXT_EXPRESSION, observed)

    def test_screenshot_is_bounded_and_only_returned_as_base64_attachment_data(self):
        module = types.SimpleNamespace(InvisiblePlaywright=FakeBrowser)
        with patch.dict(sys.modules, {"invisible_playwright": module}), patch.object(browser, "_LoopbackFilteringProxy", FakeFilteringProxy):
            result = browser.run_probe({"url": "https://example.com", "screenshot": True})
        self.assertEqual(base64.b64decode(result["screenshot_base64"]), b"jpeg-image")
        self.assertNotIn("fake-proxy-secret", json.dumps(result))

    def test_oversized_screenshot_is_omitted_without_reading_it(self):
        original = FakePage.screenshot

        def large_screenshot(self, path, **kwargs):
            with open(path, "wb") as handle:
                handle.write(b"x" * 20)

        module = types.SimpleNamespace(InvisiblePlaywright=FakeBrowser)
        with patch.dict(sys.modules, {"invisible_playwright": module}), patch.object(browser, "_LoopbackFilteringProxy", FakeFilteringProxy), patch.object(browser, "MAX_SCREENSHOT_BYTES", 8), patch.object(FakePage, "screenshot", large_screenshot):
            result = browser.run_probe({"url": "https://example.com", "screenshot": True})
        self.assertIn("screenshot_omitted", result)
        self.assertNotIn("screenshot_base64", result)
        FakePage.screenshot = original

    def test_exception_secrets_never_reach_helper_stdout(self):
        class Stdin:
            buffer = io.BytesIO(b'{"url":"https://example.com"}\n')

        output = io.StringIO()
        with patch.object(browser, "run_probe", side_effect=RuntimeError("proxy-password=fake-proxy-secret local=/secret/path")), patch.object(browser.sys, "stdin", Stdin()), patch.object(browser.sys, "stdout", output):
            browser.main()
        rendered = output.getvalue()
        self.assertNotIn("fake-proxy-secret", rendered)
        self.assertNotIn("/secret/path", rendered)
        self.assertEqual(json.loads(rendered)["status"], "ERROR")

    def test_malformed_json_has_a_safe_error_response(self):
        class Stdin:
            buffer = io.BytesIO(b'{"bad":\n')

        output = io.StringIO()
        with patch.object(browser.sys, "stdin", Stdin()), patch.object(browser.sys, "stdout", output):
            browser.main()
        self.assertEqual(json.loads(output.getvalue())["code"], "invalid_request")

    def test_request_size_is_bounded(self):
        class Stdin:
            buffer = io.BytesIO(b" " * (browser.MAX_REQUEST_BYTES + 1))

        output = io.StringIO()
        with patch.object(browser.sys, "stdin", Stdin()), patch.object(browser.sys, "stdout", output):
            browser.main()
        self.assertEqual(json.loads(output.getvalue())["code"], "invalid_request")

    def test_browser_process_group_cleans_descendants(self):
        child_code = "import subprocess,time; p=subprocess.Popen(['sleep','60']); print(p.pid,flush=True); time.sleep(60)"
        parent = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        descendant_pid = int(parent.stdout.readline().strip())
        os.killpg(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        parent.stdout.close()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{descendant_pid}/stat").read_text().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.05)
        else:
            self.fail("fixture descendant survived its helper process-group termination")

    def test_plugin_uses_private_process_group_and_never_forwards_stderr_or_image_text(self):
        source = Path(__file__).resolve().parents[1] / "opencode-plugin" / "orchestrator-supervisor.ts"
        plugin = source.read_text(encoding="utf-8")
        browser_start = plugin.index("async function runBrowserInvestigate")
        browser_end = plugin.index("// Keep the bridge's complete state", browser_start)
        browser_code = plugin[browser_start:browser_end]
        self.assertIn("detached: true", browser_code)
        self.assertIn('process.kill(-pid, signal)', plugin)
        self.assertIn("browserHelperGroups", plugin)
        self.assertIn("child.stderr.resume()", browser_code)
        self.assertNotIn("stderr.slice", browser_code)
        self.assertIn("delete report.screenshot_base64", browser_code)
        self.assertIn("network_requests_checked", browser_code)
        self.assertNotIn("Buffer.from(encodedImage, \"base64\")", browser_code)
        self.assertIn("V2_INVISIBLE_BROWSER_PROXY_CONFIG", plugin)


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp is required for loopback proxy tests")
class LoopbackFilteringProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream_requests = []
        self.upstream_tasks = set()
        self.tls_temp = tempfile.TemporaryDirectory()
        cert_path = Path(self.tls_temp.name) / "proxy.crt"
        key_path = Path(self.tls_temp.name) / "proxy.key"
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-subj", "/CN=proxy.test", "-addext", "subjectAltName=DNS:proxy.test",
                "-keyout", str(key_path), "-out", str(cert_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        server_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_tls.load_cert_chain(certfile=cert_path, keyfile=key_path)
        client_tls = ssl.create_default_context(cafile=str(cert_path))
        self.ssl_patch = patch.object(browser, "_proxy_ssl_context", return_value=client_tls)
        self.ssl_patch.start()
        self.addCleanup(self.ssl_patch.stop)
        self.addCleanup(self.tls_temp.cleanup)

        async def fake_upstream(reader, writer):
            task = asyncio.current_task()
            self.upstream_tasks.add(task)
            try:
                line = (await reader.readline()).decode("latin-1").strip()
                headers = {}
                while True:
                    value = await reader.readline()
                    if value in (b"\r\n", b"\n", b""):
                        break
                    name, _, content = value.partition(b":")
                    headers[name.decode("latin-1").lower()] = content.decode("latin-1").strip()
                self.upstream_requests.append((line, headers))
                if line.startswith("CONNECT "):
                    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    await writer.drain()
                    while data := await reader.read(4096):
                        writer.write(data)
                        await writer.drain()
                elif "/fixture-redirect" in line:
                    target = f"http://127.0.0.1:{self.private_port}/redirected"
                    response = f"HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode()
                    writer.write(response)
                    await writer.drain()
                elif "/redirect" in line:
                    writer.write(b"HTTP/1.1 302 Found\r\nLocation: http://100.85.27.103/private\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    await writer.drain()
                elif "/download" in line:
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Disposition: attachment; filename=x.bin\r\nContent-Length: 4\r\nConnection: close\r\n\r\nDATA")
                    await writer.drain()
                elif "/fixture" in line:
                    private = self.private_port
                    body = (
                        f'<html><body>fixture<img src="http://127.0.0.1:{private}/image">'
                        '<iframe src="http://example.test/fixture-redirect"></iframe>'
                        f'<script>fetch("http://127.0.0.1:{private}/fetch").catch(()=>{{}});'
                        f'let x=new XMLHttpRequest();x.open("GET","http://127.0.0.1:{private}/xhr");'
                        f'x.send();new Image().src="http://127.0.0.1:{private}/img2";'
                        f'try{{new WebSocket("ws://127.0.0.1:{private}/ws")}}catch(e){{}}</script>'
                        "</body></html>"
                    ).encode()
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
                    await writer.drain()
                else:
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                    await writer.drain()
            except (ConnectionError, ssl.SSLError):
                pass
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, ssl.SSLError):
                    pass
                self.upstream_tasks.discard(task)

        self.upstream_server = await asyncio.start_server(fake_upstream, "127.0.0.1", 0, ssl=server_tls)
        upstream_port = self.upstream_server.sockets[0].getsockname()[1]
        self.filter = browser._LoopbackFilteringProxy({
            "server": f"https://proxy.test:{upstream_port}",
            "hostname": "proxy.test",
            "pinned_ip": "127.0.0.1",
            "username": "worker-user",
            "password": "fake-proxy-secret_session-test1234_lifetime-10m",
        })
        original_resolve = browser._resolve_public

        def resolve_for_fixture(hostname, port):
            try:
                address = browser.ipaddress.ip_address(hostname)
            except ValueError:
                return {GLOBAL}
            if not browser._global_address(str(address)):
                raise browser.BrowserPolicyError("target_rejected", "browser target rejected by network policy")
            return original_resolve(hostname, port)

        self.dns_patch = patch.object(browser, "_resolve_public", side_effect=resolve_for_fixture)
        self.dns_patch.start()
        self.addCleanup(self.dns_patch.stop)
        self.filter.__enter__()

    async def asyncTearDown(self):
        self.filter.close()
        self.upstream_server.close()
        await self.upstream_server.wait_closed()
        if self.upstream_tasks:
            await asyncio.wait_for(asyncio.gather(*self.upstream_tasks, return_exceptions=True), timeout=2)

    async def request(self, request_line: str) -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.filter._port)
        settings = self.filter.browser_settings
        local_auth = base64.b64encode(f"{settings['username']}:{settings['password']}".encode()).decode()
        writer.write((request_line + f"\r\nHost: example.test\r\nProxy-Authorization: Basic {local_auth}\r\nConnection: close\r\n\r\n").encode("ascii"))
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=5)
        writer.close()
        await writer.wait_closed()
        return response

    async def test_allowed_http_goes_only_to_upstream_with_auth_and_no_auth_returns(self):
        response = await self.request("GET http://example.test/ HTTP/1.1")
        self.assertIn(b"200 OK", response)
        self.assertTrue(response.endswith(b"ok"))
        self.assertNotIn(b"fake-proxy-secret", response)
        self.assertEqual(len(self.upstream_requests), 1)
        self.assertTrue(self.upstream_requests[0][0].startswith(f"GET http://{GLOBAL}/"))
        self.assertEqual(self.upstream_requests[0][1]["host"], "example.test")
        auth = self.upstream_requests[0][1]["proxy-authorization"]
        self.assertRegex(
            base64.b64decode(auth.split()[1]).decode(),
            r"^worker-user:fake-proxy-secret_session-test1234_lifetime-10m$",
        )

    async def test_redirect_to_private_is_rejected_before_upstream_contact(self):
        first = await self.request("GET http://example.test/redirect HTTP/1.1")
        self.assertIn(b"302 Found", first)
        second = await self.request("GET http://100.85.27.103/private HTTP/1.1")
        self.assertIn(b"403 Forbidden", second)
        self.assertEqual(len(self.upstream_requests), 1)

    async def test_private_websocket_connect_is_rejected_before_upstream_contact(self):
        response = await self.request("CONNECT 100.85.27.103:443 HTTP/1.1")
        self.assertIn(b"403 Forbidden", response)
        self.assertEqual(self.upstream_requests, [])

    async def test_attachment_response_is_blocked_before_browser_delivery(self):
        response = await self.request("GET http://example.test/download HTTP/1.1")
        self.assertIn(b"403 Forbidden", response)
        self.assertNotIn(b"DATA", response)

    async def test_loopback_proxy_requires_per_run_authentication(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.filter._port)
        writer.write(b"GET http://example.test/ HTTP/1.1\r\nHost: example.test\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=5)
        writer.close()
        await writer.wait_closed()
        self.assertIn(b"407 Proxy Authentication Required", response)
        self.assertEqual(self.upstream_requests, [])

    async def test_https_tunnel_uses_validated_ip_while_tls_sni_remains_in_browser_tunnel(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.filter._port)
        settings = self.filter.browser_settings
        local_auth = base64.b64encode(f"{settings['username']}:{settings['password']}".encode()).decode()
        writer.write((f"CONNECT example.test:443 HTTP/1.1\r\nHost: example.test:443\r\nProxy-Authorization: Basic {local_auth}\r\n\r\n").encode())
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), timeout=5)
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass
        writer.close()
        await writer.wait_closed()
        self.assertIn(b"200 Connection Established", status)
        self.assertTrue(self.upstream_requests[0][0].startswith(f"CONNECT {GLOBAL}:443"))

    @unittest.skipUnless(
        os.environ.get("V2_BROWSER_ENGINE_INTEGRATION") == "1"
        and importlib.util.find_spec("invisible_playwright"),
        "set V2_BROWSER_ENGINE_INTEGRATION=1 on Cortana to test the installed browser engine",
    )
    async def test_installed_browser_routes_private_page_requests_through_the_filter(self):
        private_hits = []

        async def private_target(reader, writer):
            private_hits.append((await reader.readline()).decode("latin-1").strip())
            while await reader.readline() not in (b"\r\n", b"\n", b""):
                pass
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\nx")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        trap = await asyncio.start_server(private_target, "127.0.0.1", 0)
        self.private_port = trap.sockets[0].getsockname()[1]
        try:
            original_resolve = browser._resolve_public
            with patch.object(browser, "_resolve_public", side_effect=lambda host, port: {GLOBAL} if host == "example.test" else original_resolve(host, port)):
                def browse_fixture():
                    from invisible_playwright import InvisiblePlaywright
                    import invisible_playwright.launcher as launcher
                    from invisible_core._geo import SessionGeo

                    # The fake upstream cannot answer the package's external egress-IP
                    # discovery endpoints. Stub only that unrelated startup probe; all
                    # browser page traffic still traverses the real local filter below.
                    with patch.object(launcher, "prepare_session_geo", return_value=SessionGeo("UTC", None)):
                        with InvisiblePlaywright(
                            headless=True,
                            proxy=self.filter.browser_settings,
                            locale="en-US",
                            extra_prefs=browser.FIREFOX_NETWORK_PREFS,
                        ) as browser_instance:
                            context = browser_instance.new_context(service_workers="block", accept_downloads=False)
                            try:
                                page = context.new_page()
                                page.goto("http://example.test/fixture", timeout=15_000, wait_until="load")
                                time.sleep(0.5)
                                return page.evaluate("'serviceWorker' in navigator")
                            finally:
                                context.close()

                self.assertFalse(await asyncio.to_thread(browse_fixture))
            self.assertEqual(private_hits, [])
            self.assertGreater(self.filter.checked_requests, 1)
        finally:
            trap.close()
            await trap.wait_closed()


if __name__ == "__main__":
    unittest.main()
