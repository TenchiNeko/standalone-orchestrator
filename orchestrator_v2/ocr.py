from __future__ import annotations

import json
import subprocess
from pathlib import Path


class OCRDelegation:
    def __init__(self, executable: str = "ocr"):
        self.executable = executable

    def preview(self, repo: Path) -> dict:
        p = subprocess.run([self.executable, "delegate", "preview", "--repo", str(repo), "--format", "json"], capture_output=True, text=True, timeout=30)
        if p.returncode:
            raise RuntimeError(p.stderr.strip() or p.stdout.strip())
        try: return json.loads(p.stdout)
        except json.JSONDecodeError: return {"raw": p.stdout}

    def rules(self, repo: Path, files: list[str]) -> str:
        p = subprocess.run([self.executable, "delegate", "rule", "--repo", str(repo), *files], capture_output=True, text=True, timeout=30)
        if p.returncode: raise RuntimeError(p.stderr.strip() or p.stdout.strip())
        return p.stdout
