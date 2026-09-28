"""Extractive answering: no LLM, so nothing can be invented.

Selects the source sentences that best cover the question and returns them verbatim
with citations. Used when no LLM is configured, when the LLM call fails, and in CI so
the quality gate measures the pipeline deterministically.
"""

from __future__ import annotations

from citerx.models import ScoredChunk
from citerx.text import sentences, stems

NO_ANSWER = "The provided documents do not answer this question."
_SAFETY_WORDS = ("warning", "contraindicat", "caution", "do not", "avoid", "allerg")


def _cite(sentence: str, n: int) -> str:
    sentence = sentence.rstrip()
    if sentence and sentence[-1] in ".!?":
        return f"{sentence[:-1]} [{n}]{sentence[-1]}"
    return f"{sentence} [{n}]."


def extractive_answer(question: str, sources: list[ScoredChunk], max_sentences: int = 7, min_score: float = 0.34) -> str:
    q = set(stems(question))
    if not q:
        return NO_ANSWER

    # Safety sentences get a bonus only where they belong to the topic: the top-ranked
    # section, or a warning the retriever pulled in for that section.
    top = sources[0].chunk

    def topical(src: ScoredChunk) -> bool:
        same = (src.chunk.doc_id, src.chunk.section) == (top.doc_id, top.section)
        return same or bool(src.signals.get("added_for_safety"))

    scored: list[tuple[float, int, int, str]] = []
    for n, src in enumerate(sources, start=1):
        rank_weight = 1.0 / (1 + 0.25 * (n - 1))
        section = set(stems(src.chunk.section))
        prev: set[str] = set()
        for pos, sent in enumerate(sentences(src.chunk.text)):
            toks = set(stems(sent))
            if len(toks) < 3:
                prev = toks
                continue
            # Half credit for words in the section title ("Discharge Medications") and in
            # the previous sentence (resolves "Stop them ..." after "SGLT2 inhibitors ...").
            context = (section | prev) - toks
            overlap = (len(q & toks) + 0.5 * len(q & context)) / len(q)
            prev = toks
            if overlap == 0:
                continue
            safety = 0.15 if topical(src) and any(w in sent.lower() for w in _SAFETY_WORDS) else 0.0
            scored.append((overlap * rank_weight + safety, n, pos, sent))

    if not scored:
        return NO_ANSWER
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][0]
    if best < min_score + 0.1:
        return NO_ANSWER

    cutoff = max(min_score, 0.6 * best)
    picked = [s for s in scored if s[0] >= cutoff][:max_sentences]
    picked.sort(key=lambda x: (x[1], x[2]))  # document order reads naturally
    return " ".join(_cite(sent, n) for _, n, _, sent in picked)
