"""Settings, read from environment variables (prefix CITERX_) when Settings() is created."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PKG_PARENT = Path(__file__).resolve().parent.parent
# In a source checkout, paths are relative to the repo; when pip-installed, to the working directory.
ROOT = _PKG_PARENT if (_PKG_PARENT / "samples").is_dir() else Path.cwd()


def _env(name: str, default: str, cast=str):
    return field(default_factory=lambda: cast(os.environ.get(f"CITERX_{name}", default)))


@dataclass
class Settings:
    # Storage
    index_dir: Path = _env("INDEX_DIR", str(ROOT / "data" / "index"), Path)

    # Chunking
    max_chunk_chars: int = _env("MAX_CHUNK_CHARS", "1100", int)
    min_chunk_chars: int = _env("MIN_CHUNK_CHARS", "200", int)

    # Retrieval funnel: candidates from each retriever → fused → final context
    candidates_per_retriever: int = _env("CANDIDATES", "15", int)
    fused_k: int = _env("FUSED_K", "10", int)
    final_k: int = _env("FINAL_K", "5", int)
    rrf_k: int = 60

    # Models: "auto" uses the neural model when sentence-transformers is installed,
    # otherwise a lightweight fallback (see index/embedder.py and retrieve/reranker.py).
    embedder: str = _env("EMBEDDER", "auto")
    embed_model: str = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker: str = _env("RERANKER", "auto")
    rerank_model: str = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM: any OpenAI-compatible endpoint (Ollama, Groq, OpenRouter, vLLM ...). Empty = extractive mode.
    llm_base_url: str = _env("LLM_BASE_URL", "")
    llm_api_key: str = _env("LLM_API_KEY", "ollama")
    llm_model: str = _env("LLM_MODEL", "qwen3:latest")
    llm_timeout: int = _env("LLM_TIMEOUT", "120", int)

    # Claim verification thresholds (share of a claim's content words found in its source)
    supported_threshold: float = _env("SUPPORTED_THRESHOLD", "0.6", float)
    weak_threshold: float = _env("WEAK_THRESHOLD", "0.35", float)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url)


def get_settings() -> Settings:
    return Settings()
