"""Evaluate the frozen local NanoJev set without changing controller authority.

The deterministic baseline is deliberately conservative: it chooses the
safe/unknown answer for each question family. The Qwen baseline is one short
classification request per final-set example and is optional because it uses
the already-running local endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator_v2.decisions import NanoJevProvider  # noqa: E402
from orchestrator_v2.qwen import QwenAdapter  # noqa: E402

SAFE_DEFAULTS = {
    "operation_result": "UNCERTAIN",
    "progress_status": "NO_NEW_EVIDENCE",
    "failure_source": "UNKNOWN",
    "hypothesis_status": "INSUFFICIENT_EVIDENCE",
    "next_investigation": "ESCALATE",
    "review_finding": "INSUFFICIENT_EVIDENCE",
    "completion_support": "EVIDENCE_INCOMPLETE",
}
DANGEROUS = {
    ("operation_result", "UNCERTAIN", "SUCCESS"),
    ("progress_status", "NO_NEW_EVIDENCE", "NEW_EVIDENCE"),
    ("review_finding", "UNSUPPORTED", "SUPPORTED"),
    ("completion_support", "EVIDENCE_INCOMPLETE", "EVIDENCE_SUFFICIENT"),
}


def load_rows(path: Path):
    rows = json.loads(path.read_text())
    if len(rows) != 105 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("frozen dataset must contain 105 unique rows")
    return rows


def expected(rows):
    return [r for r in rows if r["split"] == "final"]


def score(rows, predictions, distributions=None):
    by_family = defaultdict(lambda: {"correct": 0, "total": 0})
    dangerous = []
    for row, pred in zip(rows, predictions):
        fam = by_family[row["family"]]
        fam["total"] += 1
        if pred == row["gold"]:
            fam["correct"] += 1
        if (row["family"], pred, row["gold"]) in DANGEROUS:
            dangerous.append(row["id"])
    result = {
        "n": len(rows),
        "accuracy": sum(p == r["gold"] for r, p in zip(rows, predictions)) / len(rows),
        "per_category": {k: {**v, "accuracy": v["correct"] / v["total"]} for k, v in by_family.items()},
        "dangerous_errors": dangerous,
    }
    if distributions:
        brier = []
        ece_bins = defaultdict(lambda: [0, 0, 0.0])
        for row, pred, dist in zip(rows, predictions, distributions):
            candidates = row["candidates"]
            probs = [float(dist.get(c, 0.0)) for c in candidates]
            total = sum(probs)
            if total <= 0:
                continue
            probs = [p / total for p in probs]
            brier.append(sum((p - (c == row["gold"])) ** 2 for p, c in zip(probs, candidates)))
            confidence = max(probs)
            correct = int(candidates[max(range(len(probs)), key=probs.__getitem__)] == row["gold"])
            bucket = min(9, int(confidence * 10))
            ece_bins[bucket][0] += 1; ece_bins[bucket][1] += correct; ece_bins[bucket][2] += confidence
        result["brier"] = statistics.mean(brier) if brier else None
        result["ece"] = sum(abs((v[1] / v[0]) - (v[2] / v[0])) * v[0] / len(rows) for v in ece_bins.values()) if ece_bins else None
    return result


def run_nanojev(rows, checkpoint: Path):
    provider = NanoJevProvider(checkpoint)
    predictions, distributions, latencies = [], [], []
    start = time.perf_counter()
    for row in rows:
        question = {"q": {"type": "choice", "instructions": row["question"], "criteria": row["candidate_descriptions"]}}
        batch = provider.decide(row["state"], question)
        d = batch.decisions[0]
        predictions.append(d.selected)
        distributions.append(d.distribution)
        latencies.append(d.latency)
    return {"predictions": predictions, "distributions": distributions, "latencies": latencies, "elapsed": time.perf_counter() - start, "calls": provider.calls}


def parse_qwen(content: str, candidates: list[str]) -> str | None:
    text = content.upper()
    # Prefer a JSON-like answer, then a standalone candidate. Do not treat
    # arbitrary prose as a successful structured classification.
    m = re.search(r'"(?:choice|answer|selected)"\s*:\s*"?([A-Z_]+)', text)
    if m and m.group(1) in candidates:
        return m.group(1)
    for candidate in candidates:
        if re.search(rf"(?<![A-Z_]){re.escape(candidate)}(?![A-Z_])", text):
            return candidate
    return None


def run_qwen(rows):
    key = os.environ.get("QWEN_API_KEY")
    key_file = os.environ.get("QWEN_API_KEY_FILE")
    if not key and key_file:
        key = Path(key_file).read_text().strip()
    adapter = QwenAdapter(base_url=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:18089/v1"), model=os.environ.get("QWEN_MODEL", "orca27b-ultra-q6-mtp"), api_key=key, timeout=90)
    predictions, raw, usage, elapsed = [], [], [], []
    start = time.perf_counter()
    for row in rows:
        prompt = {
            "state": row["state"], "question": row["question"],
            "candidates": row["candidates"],
            "instruction": "Return exactly one JSON object {\"choice\":\"CANDIDATE\"}; do not explain.",
        }
        response = adapter.chat([{"role": "system", "content": "Classify the supplied fixture. This is a bounded label task; do not use tools or reason aloud."}, {"role": "user", "content": json.dumps(prompt)}], max_tokens=80)
        predictions.append(parse_qwen(response.content, row["candidates"]))
        raw.append(response.content[:500]); usage.append(response.usage); elapsed.append(response.elapsed)
    return {"predictions": predictions, "raw": raw, "usage": usage, "latencies": elapsed, "elapsed": time.perf_counter() - start, "calls": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path(__file__).with_name("nanojev_eval.json")))
    ap.add_argument("--checkpoint", default=os.environ.get("NANOJEV_CHECKPOINT", "/home/brandon/models/nanojev"))
    ap.add_argument("--qwen", action="store_true", help="also run the bounded local Qwen final-set baseline")
    ap.add_argument("--output", default=str(ROOT / ".orchestrator-v2" / "nanojev-evaluation.json"))
    args = ap.parse_args()
    all_rows = load_rows(Path(args.dataset)); final_rows = expected(all_rows)
    result = {"dataset": {"total": len(all_rows), "development": 70, "final": len(final_rows), "split": "10 development + 5 untouched final per family"}}
    baseline_preds = [SAFE_DEFAULTS[row["family"]] for row in final_rows]
    result["deterministic_final"] = score(final_rows, baseline_preds)
    nano = run_nanojev(all_rows, Path(args.checkpoint))
    result["nanojev_all"] = score(all_rows, nano["predictions"], nano["distributions"])
    final_indexes = [i for i, row in enumerate(all_rows) if row["split"] == "final"]
    final_preds = [nano["predictions"][i] for i in final_indexes]
    final_dist = [nano["distributions"][i] for i in final_indexes]
    result["nanojev_final"] = score(final_rows, final_preds, final_dist)
    result["nanojev_runtime"] = {"elapsed": nano["elapsed"], "calls": nano["calls"], "p50": statistics.median(nano["latencies"]), "p95": sorted(nano["latencies"])[int(len(nano["latencies"]) * .95) - 1], "max": max(nano["latencies"])}
    if args.qwen:
        qwen = run_qwen(final_rows)
        result["qwen_final"] = score(final_rows, qwen["predictions"])
        result["qwen_runtime"] = {"elapsed": qwen["elapsed"], "calls": qwen["calls"], "p50": statistics.median(qwen["latencies"]), "p95": sorted(qwen["latencies"])[int(len(qwen["latencies"]) * .95) - 1], "usage": qwen["usage"], "unparsed": sum(p is None for p in qwen["predictions"])}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
