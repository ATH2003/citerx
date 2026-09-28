"""Command-line interface.

    citerx ingest samples/*.md docs/guideline.pdf
    citerx ask "What are the renal contraindications for metformin?"
    citerx list
    citerx clear
    citerx eval
"""

from __future__ import annotations

import argparse
import sys

STATUS_ICON = {"supported": "✔", "weak": "~", "unsupported": "✘", "uncited": "?"}


def _ask(rx, question: str, show_sources: bool):
    ans = rx.ask(question)
    if rx.last_error:
        print(f"[note] {rx.last_error}", file=sys.stderr)
    print(f"\n{ans.text}\n")
    if ans.claims:
        print(f"Claim check ({ans.mode} mode) — groundedness {ans.groundedness:.0%}")
        for c in ans.claims:
            extra = f"  numbers not in source: {', '.join(c.missing_numbers)}" if c.missing_numbers else ""
            print(f"  {STATUS_ICON[c.status]} {c.status:<11} {c.sentence[:90]}{extra}")
    if show_sources:
        print("\nSources")
        for n, s in enumerate(ans.sources, 1):
            tag = "  (added for safety)" if s.signals.get("added_for_safety") else ""
            print(f"  [{n}] {s.chunk.source} · p.{s.chunk.page} · {s.chunk.section or '—'}{tag}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="citerx", description="Cited, claim-checked answers from clinical documents.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="add documents (.pdf, .txt, .md) to the index")
    p_ing.add_argument("paths", nargs="+")
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--sources", action="store_true", help="list the retrieved sources")
    sub.add_parser("list", help="list indexed documents")
    sub.add_parser("clear", help="remove all documents from the index")
    sub.add_parser("eval", help="run the evaluation suite and quality gate")
    args = ap.parse_args(argv)

    if args.cmd == "eval":
        from citerx.eval.run import main as run_eval

        return run_eval([])

    from citerx.pipeline import CiteRx

    rx = CiteRx()
    if args.cmd == "ingest":
        for path in args.paths:
            chunks = rx.ingest(path)
            warnings = sum(c.has_warning for c in chunks)
            print(f"Indexed {path}: {len(chunks)} chunks ({warnings} with safety warnings)")
    elif args.cmd == "ask":
        if not rx.index.chunks:
            print("The index is empty. Run `citerx ingest <file>` first (try: citerx ingest samples/*).")
            return 1
        _ask(rx, " ".join(args.question), args.sources)
    elif args.cmd == "list":
        for source, chunks in rx.index.documents.items():
            print(f"{source}: {len(chunks)} chunks")
        print(rx.describe())
    elif args.cmd == "clear":
        rx.index.clear()
        print("Index cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
