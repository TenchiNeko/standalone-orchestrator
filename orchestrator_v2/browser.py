from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Screenshot:
    path: str
    sha256: str
    viewport: dict[str, int]
    url: str


class BrowserTools:
    """Disposable Playwright bridge; the browser profile is supplied by the caller."""

    def __init__(self, workspace: Path, playwright_package: str | None = None):
        self.workspace = workspace.resolve()
        self.playwright_package = playwright_package or os.environ.get("PLAYWRIGHT_PACKAGE")

    def screenshot(self, url: str, relative_path: str, viewport: dict[str, int] | None = None) -> Screenshot:
        if not self.playwright_package:
            raise RuntimeError("PLAYWRIGHT_PACKAGE is required for browser capture")
        viewport = viewport or {"width": 1280, "height": 800}
        out = (self.workspace / relative_path).resolve()
        if self.workspace not in out.parents:
            raise ValueError("screenshot path escapes workspace")
        script = """const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE); (async()=>{const b=await chromium.launch({headless:true,executablePath:process.env.BROWSER_EXECUTABLE||undefined,args:['--no-sandbox']});const p=await b.newPage({viewport:{width:Number(process.env.VW),height:Number(process.env.VH)}});await p.goto(process.env.URL,{waitUntil:'networkidle'});await p.screenshot({path:process.env.OUT});await b.close()})().catch(e=>{console.error(e);process.exit(1)})"""
        env = {"PATH": os.environ.get("PATH", ""), "PLAYWRIGHT_PACKAGE": self.playwright_package, "BROWSER_EXECUTABLE": os.environ.get("BROWSER_EXECUTABLE", "/usr/bin/google-chrome"), "URL": url, "OUT": str(out), "VW": str(viewport["width"]), "VH": str(viewport["height"])}
        subprocess.run(["node", "-e", script], env=env, cwd=self.workspace, check=True, timeout=60, capture_output=True, text=True)
        return Screenshot(relative_path, hashlib.sha256(out.read_bytes()).hexdigest(), viewport, url)


def screenshot_message(image: Screenshot) -> dict:
    """Build a metadata-only evidence record; image bytes are sent only by an explicit Qwen call."""
    return {"path": image.path, "sha256": image.sha256, "viewport": image.viewport, "url": image.url}
