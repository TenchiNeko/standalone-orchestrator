"""Small deterministic Python AST symbol index and candidate ranker."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Symbol:
    file: str
    kind: str
    name: str
    line: int
    end_line: int
    signature: str
    imports: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class SymbolCandidate:
    symbol: Symbol
    score: int
    features: tuple[str, ...] = ()


def _signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.posonlyargs + node.args.args]
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        args.extend(a.arg for a in node.args.kwonlyargs)
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)
        return f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        bases = [getattr(b, "id", getattr(b, "attr", "?")) for b in node.bases]
        return f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
    return ""


def index_file(path: Path, root: Path) -> list[Symbol]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    out: list[Symbol] = []
    rel = path.relative_to(root).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            refs = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id != node.name})
            out.append(Symbol(rel, node.__class__.__name__, node.name, getattr(node, "lineno", 0), getattr(node, "end_lineno", getattr(node, "lineno", 0)), _signature(node), tuple(sorted(set(imports))), tuple(refs)))
    return sorted(out, key=lambda s: (s.file, s.line, s.name, s.kind))


class SymbolIndex:
    def __init__(self, root: Path, permitted_files: list[str] | None = None, max_files: int = 500):
        self.root = Path(root).resolve()
        self.permitted_files = {Path(p).as_posix() for p in permitted_files} if permitted_files else None
        self.max_files = max_files

    def build(self) -> list[Symbol]:
        paths = [p for p in self.root.rglob("*.py") if ".git" not in p.parts and (self.permitted_files is None or p.relative_to(self.root).as_posix() in self.permitted_files)]
        return [symbol for path in sorted(paths)[: self.max_files] for symbol in index_file(path, self.root)]

    def candidates(self, query: str, *, limit: int = 20, recent_files: set[str] | None = None, evidence_terms: set[str] | None = None) -> list[SymbolCandidate]:
        terms = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query or "") if len(t) > 2}
        evidence_terms = {t.lower() for t in (evidence_terms or set())}
        ranked: list[SymbolCandidate] = []
        for symbol in self.build():
            blob = f"{symbol.file} {symbol.name} {symbol.signature} {' '.join(symbol.imports)} {' '.join(symbol.references)}".lower()
            matched = sorted(term for term in terms if term in blob)
            features = [f"term:{term}" for term in matched]
            score = len(matched) * 10
            if recent_files and symbol.file in recent_files:
                score += 4; features.append("recent_file")
            if evidence_terms and any(term in blob for term in evidence_terms):
                score += 3; features.append("evidence_term")
            if score:
                ranked.append(SymbolCandidate(symbol, score, tuple(features)))
        return sorted(ranked, key=lambda item: (-item.score, item.symbol.file, item.symbol.line, item.symbol.name))[:limit]

    def render(self, query: str, *, limit: int = 20, recent_files: set[str] | None = None, evidence_terms: set[str] | None = None) -> list[dict[str, object]]:
        return [{"file": c.symbol.file, "kind": c.symbol.kind, "name": c.symbol.name, "line": c.symbol.line, "end_line": c.symbol.end_line, "signature": c.symbol.signature, "score": c.score, "features": list(c.features)} for c in self.candidates(query, limit=limit, recent_files=recent_files, evidence_terms=evidence_terms)]
