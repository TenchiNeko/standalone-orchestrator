"""Bounded read-only browser evidence for the OpenCode v2 plugin."""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import secrets
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


MAX_BODY_CHARS = 1200
MAX_SELECTOR_CHARS = 200
MAX_SCREENSHOT_BYTES = 1_500_000


def _public_url(value: str, same_origin: str | None = None) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("url is required and must be bounded")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("url credentials are not permitted")
    if same_origin:
        parent = urlparse(same_origin)
        origin = (parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        parent_origin = (parent.scheme, parent.hostname, parent.port or (443 if parent.scheme == "https" else 80))
        if origin != parent_origin:
            raise ValueError("secondary navigation must stay on the original origin")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port, type=socket.SOCK_STREAM)}
    except OSError as exc:
        raise ValueError("url host could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValueError("private or local browser targets are not permitted")
    return value


def _trusted_proxy() -> dict[str, str] | None:
    config_name = os.environ.get("V2_INVISIBLE_BROWSER_PROXY_CONFIG", "").strip()
    if not config_name:
        return None
    config_path = Path(config_name).expanduser()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    proxy = config.get("proxy") if isinstance(config, dict) else None
    if not isinstance(proxy, dict) or not all(proxy.get(key) for key in ("host", "port", "user", "pass")):
        raise ValueError("configured browser proxy is incomplete")
    return {
        "server": f"{proxy['host']}:{proxy['port']}",
        "username": str(proxy["user"]),
        "password": f"{proxy['pass']}_session-{secrets.token_hex(4)}_lifetime-10m",
    }


def _text(value: object, limit: int = MAX_BODY_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def run_probe(request: dict[str, object]) -> dict[str, object]:
    url = _public_url(str(request.get("url") or ""))
    navigate_to = request.get("navigate_to")
    if navigate_to is not None:
        navigate_to = _public_url(str(navigate_to), same_origin=url)
    selector = request.get("selector")
    if selector is not None:
        selector = str(selector)
        if not selector or len(selector) > MAX_SELECTOR_CHARS or "javascript:" in selector.lower():
            raise ValueError("selector must be a bounded CSS selector")
    screenshot = bool(request.get("screenshot", True))
    from invisible_playwright import InvisiblePlaywright

    with InvisiblePlaywright(headless=True, proxy=_trusted_proxy()) as browser:
        page = browser.new_page()
        page.goto(url, timeout=60_000, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("load", timeout=20_000)
        except Exception:
            pass
        if navigate_to:
            page.goto(navigate_to, timeout=60_000, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("load", timeout=20_000)
            except Exception:
                pass
        result: dict[str, object] = {
            "status": "OK",
            "url": page.url,
            "title": _text(page.title(), 240),
            "ready_state": _text(page.evaluate("document.readyState"), 40),
            "body_text": _text(page.evaluate("document.body ? document.body.innerText : ''")),
            "webdriver": page.evaluate("navigator.webdriver"),
            "proxy_configured": bool(os.environ.get("V2_INVISIBLE_BROWSER_PROXY_CONFIG", "").strip()),
        }
        if selector:
            result["selector_text"] = _text(page.locator(selector).text_content(), 600)
        if screenshot:
            with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
                page.screenshot(path=image.name, type="jpeg", quality=75, full_page=False)
                data = Path(image.name).read_bytes()
            if len(data) <= MAX_SCREENSHOT_BYTES:
                result["screenshot_sha256"] = hashlib.sha256(data).hexdigest()
                result["screenshot_bytes"] = len(data)
                result["screenshot_base64"] = base64.b64encode(data).decode("ascii")
            else:
                result["screenshot_omitted"] = "screenshot exceeded the bounded attachment size"
    return result


def main() -> int:
    try:
        request = json.loads(sys.stdin.readline())
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        print(json.dumps(run_probe(request), ensure_ascii=False, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "reason": f"{type(exc).__name__}: {str(exc)[:240]}"}, separators=(",", ":")))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

