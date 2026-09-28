"""Evaluation harness and CI quality gate.

    python -m citerx.eval.run            # extractive mode, deterministic
    python -m citerx.eval.run --llm      # uses the configured LLM endpoint

Metrics (all 0..1, higher is better):
  hit@k           expected section appears among the retrieved sources
  mrr             reciprocal rank of the expected section
  fact_recall     share of must-include facts present in the answer
  groundedness    share of answer claims verified against their cited source
  abstention_acc  answers answerable questions, declines unanswerable ones
  safety_recall   answers to warning-bearing questions carry the warning
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

from citerx.config import ROOT, get_settings
from citerx.pipeline import CiteRx

DATASET = Path(__file__).with_name("dataset.json")
THRESHOLDS = {
    "hit@k": 0.90,
    "mrr": 0.75,
    "fact_recall": 0.80,
    "groundedness": 0.90,
    "abstention_acc": 0.90,
    "safety_recall": 0.80,
}
_WARNING = re.compile(r"warning|caution|contraindicat|do not|avoid|allerg|risk", re.I)


def evaluate(use_llm: bool = False, docs_dir: Path = ROOT / "samples") -> dict:
    settings = get_settings()
    settings.index_dir = Path(tempfile.mkdtemp(prefix="citerx-eval-"))
    rx = CiteRx(settings)
    for f in sorted(docs_dir.iterdir()):
        if f.suffix.lower() in {".pdf", ".txt", ".md"}:
            rx.ingest(f)

    cases = json.loads(DATASET.read_text())
    rows = []
    for case in cases:
        start = time.perf_counter()
        ans = rx.ask(case["question"], use_llm=use_llm)
        latency = time.perf_counter() - start
        answerable = case.get("answerable", True)
        row = {"id": case["id"], "question": case["question"], "answer": ans.text, "mode": ans.mode,
               "latency_s": round(latency, 3), "answerable": answerable, "abstained": ans.abstained}

        if answerable:
            rank = next((i for i, s in enumerate(ans.sources, 1)
                         if s.chunk.source == case["source"] and s.chunk.section.startswith(case["section"])), None)
            row["rank"] = rank
            facts = case.get("must_include", [])
            row["fact_recall"] = mean(f.lower() in ans.text.lower() for f in facts) if facts else 1.0
            row["groundedness"] = ans.groundedness
            row["flagged"] = [c.sentence for c in ans.flagged]
            if case.get("requires_warning"):
                row["has_warning"] = bool(_WARNING.search(ans.text))
        rows.append(row)

    ans_rows = [r for r in rows if r["answerable"]]
    warn_rows = [r for r in ans_rows if "has_warning" in r]
    metrics = {
        "hit@k": mean(r["rank"] is not None for r in ans_rows),
        "mrr": mean(1 / r["rank"] if r["rank"] else 0 for r in ans_rows),
        "fact_recall": mean(r["fact_recall"] for r in ans_rows),
        "groundedness": mean(r["groundedness"] for r in ans_rows if not r["abstained"]) if any(not r["abstained"] for r in ans_rows) else 0.0,
        "abstention_acc": mean(r["abstained"] != r["answerable"] for r in rows),
        "safety_recall": mean(r["has_warning"] for r in warn_rows) if warn_rows else 1.0,
    }
    return {
        "config": rx.describe() | {"mode": "llm" if use_llm else "extractive", "cases": len(rows)},
        "metrics": {k: round(v, 3) for k, v in metrics.items()},
        "thresholds": THRESHOLDS,
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate CiteRx and enforce the quality gate.")
    ap.add_argument("--llm", action="store_true", help="use the configured LLM instead of extractive mode")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval_results" / "latest.json")
    ap.add_argument("--no-gate", action="store_true", help="report only; never fail")
    args = ap.parse_args(argv)

    report = evaluate(use_llm=args.llm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\nCiteRx evaluation — {report['config']}")
    print(f"{'metric':<16}{'score':>8}{'gate':>8}")
    failed = []
    for name, value in report["metrics"].items():
        gate = THRESHOLDS[name]
        ok = value >= gate
        failed += [] if ok else [name]
        print(f"{name:<16}{value:>8.3f}{gate:>8.2f}  {'PASS' if ok else 'FAIL'}")

    for r in report["cases"]:
        problems = []
        if r["answerable"] and r.get("rank") is None:
            problems.append("expected section not retrieved")
        if r["abstained"] == r["answerable"]:
            problems.append("wrong abstention")
        if r.get("fact_recall", 1) < 1:
            problems.append(f"fact recall {r['fact_recall']:.2f}")
        if r.get("flagged"):
            problems.append(f"{len(r['flagged'])} flagged claim(s)")
        if problems:
            print(f"  - {r['id']}: {', '.join(problems)}")

    print(f"\nReport written to {args.out}")
    if failed and not args.no_gate:
        print(f"Quality gate FAILED: {', '.join(failed)}")
        return 1
    print("Quality gate passed." if not failed else "Quality gate skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
