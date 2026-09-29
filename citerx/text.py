"""Small text utilities used by retrieval and verification."""

from __future__ import annotations

import re

# Tokens keep clinical shapes intact: "ml/min", "1.73", "hba1c", "140/90".
_TOKEN = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*")
_NUMBER = re.compile(r"(?<![a-z0-9])\d+(?:[.,]\d+)?(?:/\d+(?:\.\d+)?)?")
# Split after sentence punctuation, or after a trailing citation marker like "[2]".
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])|(?<=\])\s+(?=[A-Z(])")
_SUFFIXES = ("ations", "ation", "ating", "ated", "ates", "ments", "ment", "ings", "ing", "ies", "ical", "ic", "ed", "es", "ly", "s")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may might more most must no not of on or should so such than that the their them
    then there these they this those to was were what when where which while who why will with would
    you your patient patients any anything something all use used""".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1 or t.isdigit()]


def stem(tok: str) -> str:
    """Tiny suffix stripper: enough to match discharged/discharge, doses/dose."""
    if tok.isdigit() or len(tok) <= 4 or not tok.isalpha():
        return tok
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            tok = tok[: -len(suf)]
            if suf in {"ed", "ing", "ings"} and tok[-1] == tok[-2] and tok[-1] not in "aeiouls":
                tok = tok[:-1]  # stopped -> stop, planning -> plan
            break
    return tok[:-1] if tok.endswith(("e", "y")) and len(tok) > 5 else tok


def stems(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]


def numbers(text: str) -> set[str]:
    """Numbers as written, normalised (commas → dots, trailing .0 dropped)."""
    out = set()
    for raw in _NUMBER.findall(text):
        out.add(raw.replace(",", ".").removesuffix(".0"))
    return out


def sentences(text: str) -> list[str]:
    parts = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-*•]\s)|\n(?=\s*\d+[.)]\s)", text):
        block = " ".join(block.split())
        if block:
            parts.extend(s.strip() for s in _SENTENCE.split(block) if s.strip())
    return parts
