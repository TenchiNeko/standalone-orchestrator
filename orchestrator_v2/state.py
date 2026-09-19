from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Phase(str, Enum):
    INTAKE = "intake"
    INSPECT = "inspect"
    PLAN = "plan"
    IMPLEMENT = "implement"
    TEST = "test"
    REVIEW = "review"
    VISUAL_VERIFY = "visual_verify"
    FINAL_VERIFY = "final_verify"
    RECONCILE = "reconcile"
    DIAGNOSE = "diagnose"
    REPAIR = "repair"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class Criterion:
    key: str
    description: str
    required: bool = True
    status: str = "pending"
    evidence: list[str] = field(default_factory=list)
    verified_source_hash: str | None = None


@dataclass
class TaskContract:
    goal: str
    permitted_files: list[str]
    permitted_actions: list[str]
    criteria: list[Criterion]
    version: int = 1
    test_command: list[str] | None = None
    visual_required: bool = False

    def validate(self) -> None:
        if not self.goal.strip() or not self.criteria:
            raise ValueError("contract requires a goal and at least one criterion")
        if not self.permitted_files or not self.permitted_actions:
            raise ValueError("contract must declare permitted files and actions")
        unknown = set(self.permitted_actions) - {"read_file", "write_file", "run_tests", "browser"}
        if unknown:
            raise ValueError(f"unknown permitted actions: {sorted(unknown)}")
        if self.test_command is not None and "run_tests" not in self.permitted_actions:
            raise ValueError("test_command requires the run_tests action")
        if self.test_command is not None and (not self.test_command or Path(self.test_command[0]).name not in {"python", "python3", "pytest"}):
            raise ValueError("test_command must start with python or pytest")
        if any(not c.description.strip() for c in self.criteria):
            raise ValueError("criteria must be testable, non-empty descriptions")
        if len({c.key for c in self.criteria}) != len(self.criteria):
            raise ValueError("criterion keys must be unique")


@dataclass
class Task:
    task_id: str
    contract: TaskContract
    workspace: str
    phase: Phase = Phase.INTAKE
    attempts: int = 0
    repair_cycles: int = 0
    budget_calls: int = 0
    budget_limit: int = 12
    unresolved: list[str] = field(default_factory=list)


def source_hash(root: Path, permitted: list[str] | None = None) -> str:
    h = hashlib.sha256()
    paths = [root / p for p in permitted] if permitted else sorted(root.rglob("*"))
    for p in sorted(paths):
        if p.is_file() and ".git" not in p.parts:
            rel = p.relative_to(root).as_posix()
            h.update(rel.encode()); h.update(b"\0"); h.update(p.read_bytes())
    return h.hexdigest()


def file_manifest(root: Path, permitted: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Stable file facts used by completion evidence and stale-check detection."""
    paths = [root / p for p in permitted] if permitted else sorted(root.rglob("*"))
    manifest: dict[str, dict[str, Any]] = {}
    for p in sorted(paths):
        if p.is_file() and ".git" not in p.parts:
            rel = p.relative_to(root).as_posix()
            data = p.read_bytes()
            manifest[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return manifest


class StateStore:
    """SQLite event/state store; every mutation is transactional."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, kind TEXT NOT NULL, source_hash TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            """)

    def _db(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def save_task(self, task: Task) -> None:
        task.contract.validate()
        payload = json.dumps(asdict(task), default=lambda x: x.value if isinstance(x, Enum) else x)
        with self._db() as db:
            db.execute("INSERT INTO tasks(id,payload,updated) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,updated=excluded.updated", (task.task_id, payload, time.time()))

    def load_task(self, task_id: str) -> Task:
        with self._db() as db:
            row = db.execute("SELECT payload FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(task_id)
        d = json.loads(row[0]); c = d["contract"]
        c["criteria"] = [Criterion(**x) for x in c["criteria"]]
        d["contract"] = TaskContract(**c); d["phase"] = Phase(d["phase"])
        return Task(**d)

    def event(self, task_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._db() as db:
            db.execute("INSERT INTO events(task_id,kind,payload,created) VALUES(?,?,?,?)", (task_id, kind, json.dumps(payload), time.time()))

    def evidence_record(self, ref: str) -> dict[str, Any] | None:
        try:
            row_id = int(ref.split(":", 1)[1])
        except (ValueError, IndexError):
            return None
        with self._db() as db:
            row = db.execute("SELECT id, task_id, kind, source_hash, payload, created FROM evidence WHERE id=?", (row_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        try:
            result["payload"] = json.loads(result["payload"])
        except json.JSONDecodeError:
            result["payload"] = {}
        return result

    def current_successful_check(self, task_id: str, root: Path, permitted: list[str] | None = None) -> dict[str, Any] | None:
        """Return a passing check tied to the current source hash, if one exists."""
        digest = source_hash(root, permitted)
        with self._db() as db:
            rows = db.execute(
                "SELECT id, kind, source_hash, payload, created FROM evidence "
                "WHERE task_id=? AND kind IN ('test','final_test') ORDER BY id DESC",
                (task_id,),
            ).fetchall()
        for row in rows:
            if row["source_hash"] != digest:
                continue
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                continue
            if payload.get("exit_code") == 0 and not payload.get("timed_out"):
                return {"ref": f"evidence:{row['id']}", "kind": row["kind"], "source_hash": row["source_hash"], "payload": payload, "created": row["created"]}
        return None

    def unresolved_mutations(self, task_id: str) -> list[dict[str, Any]]:
        unresolved: list[dict[str, Any]] = []
        with self._db() as db:
            rows = db.execute("SELECT kind, payload FROM events WHERE task_id=? AND kind IN ('mutation_intent','mutation_result') ORDER BY id", (task_id,)).fetchall()
        intents: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                continue
            key = payload.get("idempotency_key")
            if row["kind"] == "mutation_intent" and key:
                intents[key] = payload
            elif row["kind"] == "mutation_result" and key:
                if payload.get("status") == "unknown":
                    unresolved.append({"idempotency_key": key, "intent": intents.get(key, {}), "result": payload})
                intents.pop(key, None)
        unresolved.extend({"idempotency_key": key, "intent": value} for key, value in intents.items())
        return unresolved

    def mutation_intent(self, task_id: str, action: str, idempotency_key: str) -> None:
        self.event(task_id, "mutation_intent", {"action": action, "idempotency_key": idempotency_key, "status": "intent"})

    def mutation_ack(self, task_id: str, idempotency_key: str, status: str, result: dict[str, Any] | None = None) -> None:
        if status not in {"acknowledged", "unknown"}:
            raise ValueError("mutation status must be acknowledged or unknown")
        self.event(task_id, "mutation_result", {"idempotency_key": idempotency_key, "status": status, "result": result or {}})

    def evidence(self, task_id: str, kind: str, root: Path, payload: dict[str, Any], permitted: list[str] | None = None) -> str:
        digest = source_hash(root, permitted)
        with self._db() as db:
            cur = db.execute("INSERT INTO evidence(task_id,kind,source_hash,payload,created) VALUES(?,?,?,?,?)", (task_id, kind, digest, json.dumps(payload), time.time()))
        ref = f"evidence:{cur.lastrowid}"
        self.event(task_id, "evidence_recorded", {"ref": ref, "kind": kind, "source_hash": digest})
        return ref

    def events(self, task_id: str) -> list[dict[str, Any]]:
        with self._db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM events WHERE task_id=? ORDER BY id", (task_id,))]

    def evidence_is_current(self, ref: str, root: Path, permitted: list[str] | None = None) -> bool:
        try: row_id = int(ref.split(":", 1)[1])
        except (ValueError, IndexError): return False
        with self._db() as db:
            row = db.execute("SELECT source_hash FROM evidence WHERE id=?", (row_id,)).fetchone()
        return bool(row and row[0] == source_hash(root, permitted))


def new_task(contract: TaskContract, workspace: Path, budget_limit: int = 12) -> Task:
    contract.validate()
    return Task(str(uuid.uuid4()), contract, str(workspace), budget_limit=budget_limit)
