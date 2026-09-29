"""Claim-level verification of a generated answer.

Every sentence in the answer is treated as a claim and checked against the passages it
cites (not against the whole context, since a correct fact behind the wrong citation is
still a broken citation). Two checks run per claim:

* Lexical support: the share of the claim's content words found in the best-matching
  one- or two-sentence window of a cited passage (with light stemming).
* Numeric grounding: every number in the claim (doses, thresholds, durations) must appear
  in a cited passage. A single invented number marks the claim unsupported, whatever
  its lexical score, because in clinical text the number is usually the whole point.

This is deliberately transparent and dependency-free, so a reviewer can see exactly
why a claim was flagged. It is a guard rail, not proof of correctness.
"""

from __future__ import annotations

import re

from citerx.models import ClaimCheck, ScoredChunk
from citerx.text import content_tokens, numbers, sentences, stems

CITATION = re.compile(r"\[(?:source\s*)?(\d+(?:\s*[,;]\s*(?:source\s*)?\d+)*)\]", re.IGNORECASE)
ABSTAIN = re.compile(r"(not (?:found|covered|mentioned|stated|available)|no information|cannot (?:find|answer)|do(?:es)? not (?:contain|mention|say|answer|cover)\b)", re.IGNORECASE)


def _stems(text: str) -> set[str]:
    return set(stems(text))


def cited_ids(sentence: str) -> list[int]:
    ids: list[int] = []
    for group in CITATION.findall(sentence):
        ids.extend(int(n) for n in re.findall(r"\d+", group))
    return ids


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", " ".join(CITATION.sub("", sentence).split()))


def _windows(text: str) -> list[str]:
    sents = sentences(text) or [text]
    return sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]


def _best_support(claim: str, passage: str) -> tuple[float, str]:
    claim_stems = _stems(claim)
    if not claim_stems:
        return 1.0, ""
    best, best_win = 0.0, ""
    for win in _windows(passage):
        score = len(claim_stems & _stems(win)) / len(claim_stems)
        if score > best:
            best, best_win = score, win
    return best, best_win


def _claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split into (sentence, citations); a sentence without a marker inherits the next
    marker in the same paragraph, since models often cite once at the end of a run."""
    out: list[tuple[str, list[int]]] = []
    for para in re.split(r"\n\s*\n", answer):
        sents = [s for s in sentences(para) if not re.match(r"^\s*(sources?|references?)\s*:", s, re.IGNORECASE)]
        pending: list[str] = []
        for s in sents:
            ids = cited_ids(s)
            if ids:
                out.extend((p, ids) for p in pending)
                pending = []
                out.append((s, ids))
            else:
                pending.append(s)
        out.extend((p, []) for p in pending)
    return out


def verify(answer: str, sources: list[ScoredChunk], supported: float = 0.6, weak: float = 0.35) -> list[ClaimCheck]:
    checks: list[ClaimCheck] = []
    for sentence, ids in _claims(answer):
        claim = strip_citations(sentence)
        if len(content_tokens(claim)) < 2 or ABSTAIN.search(claim):
            continue  # connective text or an explicit "not in the documents"
        valid = [i for i in ids if 1 <= i <= len(sources)]
        if not valid:
            checks.append(ClaimCheck(claim, ids, "uncited", 0.0))
            continue

        cited_text = "\n".join(sources[i - 1].chunk.text for i in valid)
        score, evidence = max((_best_support(claim, sources[i - 1].chunk.text) for i in valid), key=lambda x: x[0])
        missing = sorted(numbers(claim) - numbers(cited_text))

        if missing:
            status = "unsupported"
        elif score >= supported:
            status = "supported"
        elif score >= weak:
            status = "weak"
        else:
            status = "unsupported"
        checks.append(ClaimCheck(claim, valid, status, round(score, 3), evidence, missing))
    return checks
