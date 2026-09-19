from __future__ import annotations

import json
import time
import urllib.request
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class QwenResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any]
    finish_reason: str | None
    elapsed: float


class QwenAdapter:
    def __init__(self, base_url: str = "http://127.0.0.1:18089/v1", api_key: str | None = None, model: str = "orca27b-ultra-q6-mtp", timeout: int = 120):
        self.base_url = base_url.rstrip("/"); self.api_key = api_key; self.model = model; self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1200) -> QwenResponse:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        if tools: body["tools"] = tools; body["tool_choice"] = "auto"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"} if self.api_key else {"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp: data = json.load(resp)
        choice = (data.get("choices") or [{}])[0]; msg = choice.get("message") or {}
        return QwenResponse(msg.get("content") or "", msg.get("tool_calls") or [], data.get("usage") or {}, choice.get("finish_reason"), time.monotonic() - start)

    def chat_with_image(self, messages: list[dict], image: Path, max_tokens: int = 500) -> QwenResponse:
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        multimodal = list(messages) + [{"role": "user", "content": [{"type": "text", "text": "Inspect this screenshot and report only the requested visual fact."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}]}]
        return self.chat(multimodal, max_tokens=max_tokens)

    def health(self) -> dict:
        req = urllib.request.Request(self.base_url + "/models", headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
        with urllib.request.urlopen(req, timeout=10) as resp: return json.load(resp)
