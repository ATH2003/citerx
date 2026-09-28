"""Prompt construction. Sources are numbered [1]..[n] to match the verifier."""

from __future__ import annotations

from citerx.models import ScoredChunk

SYSTEM_PROMPT = """You are CiteRx, an assistant that answers questions about clinical documents.

Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. End every sentence that states a fact with its source number(s), e.g. [1] or [2, 3].
3. Copy numbers (doses, thresholds, durations, lab values) exactly as the source writes them.
4. If a source contains a warning, caution or contraindication relevant to the question, include it.
5. If the sources do not contain the answer, reply exactly: "The provided documents do not answer this question."
6. Be concise: plain sentences, no preamble, no headings, no list of sources at the end."""


def format_sources(sources: list[ScoredChunk]) -> str:
    parts = []
    for n, s in enumerate(sources, start=1):
        c = s.chunk
        header = f"[{n}] {c.source}, p.{c.page}" + (f", section: {c.section}" if c.section else "")
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def build_messages(question: str, sources: list[ScoredChunk]) -> list[dict]:
    user = f"SOURCES\n\n{format_sources(sources)}\n\nQUESTION\n{question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
