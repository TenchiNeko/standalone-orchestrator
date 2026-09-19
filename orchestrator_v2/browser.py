from __future__ import annotations

import hashlib
import json
import os
import subprocess
from urllib.parse import urlparse
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
        self._check_local_url(url); viewport = viewport or {"width": 1280, "height": 800}
        out = (self.workspace / relative_path).resolve()
        if self.workspace not in out.parents:
            raise ValueError("screenshot path escapes workspace")
        script = """const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE); (async()=>{const b=await chromium.launch({headless:true,executablePath:process.env.BROWSER_EXECUTABLE||undefined,args:['--no-sandbox']});const p=await b.newPage({viewport:{width:Number(process.env.VW),height:Number(process.env.VH)}});await p.goto(process.env.URL,{waitUntil:'networkidle'});await p.screenshot({path:process.env.OUT});await b.close()})().catch(e=>{console.error(e);process.exit(1)})"""
        env = {"PATH": os.environ.get("PATH", ""), "PLAYWRIGHT_PACKAGE": self.playwright_package, "BROWSER_EXECUTABLE": os.environ.get("BROWSER_EXECUTABLE", "/usr/bin/google-chrome"), "URL": url, "OUT": str(out), "VW": str(viewport["width"]), "VH": str(viewport["height"])}
        subprocess.run(["node", "-e", script], env=env, cwd=self.workspace, check=True, timeout=60, capture_output=True, text=True)
        return Screenshot(relative_path, hashlib.sha256(out.read_bytes()).hexdigest(), viewport, url)

    def sequence(self, url: str, steps: list[dict], viewport: dict[str, int] | None = None) -> list[dict]:
        """Run a short local-only browser sequence; screenshot paths stay in workspace."""
        if not self.playwright_package or len(steps) > 12: raise RuntimeError("Playwright and at most 12 browser steps are required")
        self._check_local_url(url); viewport = viewport or {"width": 1280, "height": 800}
        safe_steps = []
        for step in steps:
            op = step.get("op")
            if op not in {"navigate", "screenshot", "click", "click_at", "type", "inspect"}: raise ValueError("unsupported browser operation")
            item = dict(step)
            if op in {"screenshot"}:
                out = (self.workspace / item["path"]).resolve()
                if self.workspace not in out.parents: raise ValueError("screenshot path escapes workspace")
                item["path"] = str(out)
            safe_steps.append(item)
        script = """const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE); (async()=>{const b=await chromium.launch({headless:true,executablePath:process.env.BROWSER_EXECUTABLE||undefined,args:['--no-sandbox']});const p=await b.newPage({viewport:{width:Number(process.env.VW),height:Number(process.env.VH)}});await p.goto(process.env.URL,{waitUntil:'networkidle'});const out=[];for(const s of JSON.parse(process.env.STEPS)){if(s.op==='navigate'){await p.goto(s.url,{waitUntil:'networkidle'});out.push({op:s.op,url:p.url()})}else if(s.op==='screenshot'){await p.screenshot({path:s.path});out.push({op:s.op,path:s.path})}else if(s.op==='click'){await p.locator(s.selector).click();out.push({op:s.op,selector:s.selector})}else if(s.op==='click_at'){await p.mouse.click(Number(s.x),Number(s.y));out.push({op:s.op,x:s.x,y:s.y})}else if(s.op==='type'){await p.locator(s.selector).fill(String(s.text||''));out.push({op:s.op,selector:s.selector})}else if(s.op==='inspect'){out.push({op:s.op,selector:s.selector,text:await p.locator(s.selector).textContent()})}}await b.close();console.log(JSON.stringify(out))})().catch(e=>{console.error(e);process.exit(1)})"""
        env = {"PATH": os.environ.get("PATH", ""), "PLAYWRIGHT_PACKAGE": self.playwright_package, "BROWSER_EXECUTABLE": os.environ.get("BROWSER_EXECUTABLE", "/usr/bin/google-chrome"), "URL": url, "VW": str(viewport["width"]), "VH": str(viewport["height"]), "STEPS": json.dumps(safe_steps)}
        p = subprocess.run(["node", "-e", script], env=env, cwd=self.workspace, check=True, timeout=60, capture_output=True, text=True)
        return json.loads(p.stdout.strip().splitlines()[-1])

    def actionable_controls(self, url: str, viewport: dict[str, int] | None = None) -> list[dict]:
        """Return a bounded, deterministic control table; no selectors leave this method."""
        if not self.playwright_package:
            raise RuntimeError("PLAYWRIGHT_PACKAGE is required for browser controls")
        self._check_local_url(url); viewport = viewport or {"width": 1280, "height": 800}
        script = r'''const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE);(async()=>{const b=await chromium.launch({headless:true,executablePath:process.env.BROWSER_EXECUTABLE||undefined,args:['--no-sandbox']});const p=await b.newPage({viewport:{width:Number(process.env.VW),height:Number(process.env.VH)}});await p.goto(process.env.URL,{waitUntil:'networkidle'});const rows=await p.locator('button,input,textarea,select,a,[role="button"],[role="link"]').evaluateAll((els)=>els.map((e,i)=>{const r=e.getBoundingClientRect(),label=(e.getAttribute('aria-label')||e.innerText||e.getAttribute('placeholder')||e.getAttribute('name')||'').trim().replace(/\s+/g,' ').slice(0,160);const tag=e.tagName.toLowerCase();const kind=tag==='input'?(e.type||'textbox'):(tag==='a'?'link':tag);return {id:'control-'+(i+1),index:i,kind,label,enabled:!e.disabled,visible:!!(r.width&&r.height)};}).filter(x=>x.visible&&x.enabled&&x.label));await b.close();console.log(JSON.stringify(rows.slice(0,64)))})().catch(e=>{console.error(e);process.exit(1)})'''
        env = {"PATH": os.environ.get("PATH", ""), "PLAYWRIGHT_PACKAGE": self.playwright_package, "BROWSER_EXECUTABLE": os.environ.get("BROWSER_EXECUTABLE", "/usr/bin/google-chrome"), "URL": url, "VW": str(viewport["width"]), "VH": str(viewport["height"])}
        p = subprocess.run(["node", "-e", script], env=env, cwd=self.workspace, check=True, timeout=60, capture_output=True, text=True)
        return json.loads(p.stdout.strip().splitlines()[-1])

    @staticmethod
    def _check_local_url(url: str) -> None:
        if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}: raise ValueError("browser target must be local")


def screenshot_message(image: Screenshot) -> dict:
    """Build a metadata-only evidence record; image bytes are sent only by an explicit Qwen call."""
    return {"path": image.path, "sha256": image.sha256, "viewport": image.viewport, "url": image.url}
