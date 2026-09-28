"""The retrieval funnel: hybrid search → RRF → rerank → safety expansion.

Safety expansion is the clinical twist: if a selected passage comes from a section that
also contains a WARNING/contraindication chunk, that chunk is pulled into the context
even if it ranked low. A question about a dose should never be answered without the
caution that sits beside it in the source.
"""

from __future__ import annotations

from citerx.config import Settings
from citerx.index.store import DocumentIndex
from citerx.models import ScoredChunk
from citerx.retrieve.fusion import rrf
from citerx.retrieve.reranker import Reranker


class Retriever:
    def __init__(self, index: DocumentIndex, reranker: Reranker, settings: Settings):
        self.index, self.reranker, self.s = index, reranker, settings

    def retrieve(self, query: str, k: int | None = None, expand_safety: bool = True) -> list[ScoredChunk]:
        k = k or self.s.final_k
        n = self.s.candidates_per_retriever
        dense = [i for i, _ in self.index.dense_search(query, n)]
        sparse = [i for i, _ in self.index.keyword_search(query, n)]
        fused = rrf({"dense": dense, "bm25": sparse}, k=self.s.rrf_k)[: self.s.fused_k]

        candidates = [ScoredChunk(self.index.chunks[i], score, dict(sig)) for i, score, sig in fused]
        ranked = self.reranker.rerank(query, candidates)[:k]

        if expand_safety:
            ranked = self._expand_safety(ranked)
        return ranked

    def _expand_safety(self, ranked: list[ScoredChunk]) -> list[ScoredChunk]:
        chosen = {r.chunk.chunk_id for r in ranked}
        sections = {(r.chunk.doc_id, r.chunk.section) for r in ranked if r.chunk.section}
        extra = [
            ScoredChunk(c, 0.0, {"added_for_safety": True})
            for c in self.index.chunks
            if c.has_warning and c.chunk_id not in chosen and (c.doc_id, c.section) in sections
        ]
        return ranked + extra[:2]
