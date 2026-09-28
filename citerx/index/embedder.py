"""Sentence embeddings.

`auto` uses a sentence-transformers model when the package is installed. Without it,
CiteRx falls back to a hashed word + character n-gram vectoriser: not semantic, but
deterministic, fast, and good enough to keep the app and CI working with zero downloads.
The active backend is always reported, so evaluation numbers are never ambiguous.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


class Embedder:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # rows are L2-normalised
        raise NotImplementedError


class NeuralEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 2**14):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = "hashing-ngrams (fallback)"
        self.dim = dim
        self._words = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=False, norm=None)
        self._chars = HashingVectorizer(
            n_features=dim, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False, norm=None
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        m = self._words.transform(texts) + 0.5 * self._chars.transform(texts)
        m = np.log1p(m.toarray()).astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(norms, 1e-9)


@lru_cache(maxsize=4)
def get_embedder(kind: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    if kind in {"auto", "neural"}:
        try:
            return NeuralEmbedder(model_name)
        except Exception:
            if kind == "neural":
                raise
    return HashingEmbedder()
