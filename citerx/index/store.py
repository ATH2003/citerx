"""A persistent document index holding chunks, their vectors and a BM25 model.

Vectors live in a single normalised NumPy matrix and search is an exact dot product.
For clinical document sets (hundreds to tens of thousands of chunks) this is faster
than the round-trip to a vector database and has nothing to configure. The class keeps
a narrow interface (add / remove / search) so swapping in FAISS or Chroma later is local.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from citerx.index.bm25 import BM25
from citerx.index.embedder import Embedder
from citerx.models import Chunk


class DocumentIndex:
    def __init__(self, embedder: Embedder, directory: Path | None = None):
        self.embedder = embedder
        self.directory = Path(directory) if directory else None
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self._bm25: BM25 | None = None
        if self.directory:
            self._load()

    # ---------- persistence ----------
    def _paths(self):
        return self.directory / "chunks.json", self.directory / "vectors.npy", self.directory / "meta.json"

    def _load(self):
        chunks_p, vecs_p, meta_p = self._paths()
        if not chunks_p.exists():
            return
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        chunks = [Chunk.from_dict(c) for c in json.loads(chunks_p.read_text())]
        if meta.get("embedder") == self.embedder.name and vecs_p.exists():
            self.chunks, self.vectors = chunks, np.load(vecs_p)
        else:  # embedder changed since the index was built: re-embed rather than mix spaces
            self.chunks = []
            self.add(chunks)
        self._bm25 = None

    def save(self):
        if not self.directory:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        chunks_p, vecs_p, meta_p = self._paths()
        chunks_p.write_text(json.dumps([c.to_dict() for c in self.chunks], indent=1))
        np.save(vecs_p, self.vectors)
        meta_p.write_text(json.dumps({"embedder": self.embedder.name, "chunks": len(self.chunks)}))

    # ---------- mutation ----------
    def add(self, chunks: list[Chunk]):
        if not chunks:
            return
        self.remove({c.doc_id for c in chunks})  # re-ingesting a file replaces it
        vecs = self.embedder.encode([c.embed_text() for c in chunks])
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vecs]) if len(self.vectors) else vecs
        self._bm25 = None
        self.save()

    def remove(self, doc_ids: set[str]):
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id not in doc_ids]
        if len(keep) == len(self.chunks):
            return
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._bm25 = None
        self.save()

    def clear(self):
        self.remove({c.doc_id for c in self.chunks})

    # ---------- queries ----------
    @property
    def documents(self) -> dict[str, list[Chunk]]:
        docs: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            docs.setdefault(c.source, []).append(c)
        return docs

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query])[0]
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def keyword_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        if self._bm25 is None:
            self._bm25 = BM25([c.embed_text() for c in self.chunks])
        return self._bm25.top(query, k)
