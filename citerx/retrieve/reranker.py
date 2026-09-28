"""Second-stage precision reranking with a cross-encoder (optional).

When sentence-transformers isn't installed the fused RRF order is kept, lightly
adjusted by query-term coverage so passages that answer every part of the question rise.
"""

from __future__ import annotations

from functools import lru_cache

from citerx.models import ScoredChunk
from citerx.text import stems


class Reranker:
    name = "none"

    def rerank(self, query: str, items: list[ScoredChunk]) -> list[ScoredChunk]:
        return items


class CoverageReranker(Reranker):
    name = "rrf + term coverage (fallback)"

    def rerank(self, query, items):
        q = set(stems(query))
        if not q or not items:
            return items
        top = max(i.score for i in items) or 1.0
        for item in items:
            coverage = len(q & set(stems(item.chunk.embed_text()))) / len(q)
            item.signals["coverage"] = round(coverage, 3)
            item.score = 0.6 * (item.score / top) + 0.4 * coverage
        return sorted(items, key=lambda i: i.score, reverse=True)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, query, items):
        if not items:
            return items
        scores = self.model.predict([(query, i.chunk.embed_text()) for i in items])
        for item, s in zip(items, scores):
            item.signals["rerank"] = round(float(s), 3)
            item.score = float(s)
        return sorted(items, key=lambda i: i.score, reverse=True)


@lru_cache(maxsize=2)
def get_reranker(kind: str = "auto", model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    if kind in {"auto", "cross-encoder"}:
        try:
            return CrossEncoderReranker(model_name)
        except Exception:
            if kind == "cross-encoder":
                raise
    if kind == "none":
        return Reranker()
    return CoverageReranker()
