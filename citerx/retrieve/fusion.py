"""Reciprocal Rank Fusion: combine ranked lists without having to calibrate their scores."""

from __future__ import annotations


def rrf(rankings: dict[str, list[int]], k: int = 60) -> list[tuple[int, float, dict]]:
    """rankings maps a retriever name to item ids, best first.

    Returns (item, fused_score, {"<name>_rank": r}) sorted by fused score.
    """
    scores: dict[int, float] = {}
    signals: dict[int, dict] = {}
    for name, items in rankings.items():
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            signals.setdefault(item, {})[f"{name}_rank"] = rank
    ordered = sorted(scores, key=lambda i: (-scores[i], min(signals[i].values())))
    return [(i, scores[i], signals[i]) for i in ordered]
