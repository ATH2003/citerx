"""The public entry point: `CiteRx().ask(question)` → a verified `Answer`."""

from __future__ import annotations

from pathlib import Path

from citerx.config import Settings, get_settings
from citerx.generate.extractive import NO_ANSWER, extractive_answer
from citerx.generate.llm import LLMError, chat
from citerx.generate.prompts import build_messages
from citerx.index.embedder import get_embedder
from citerx.index.store import DocumentIndex
from citerx.ingest.chunker import chunk_file
from citerx.models import Answer, Chunk
from citerx.retrieve.reranker import get_reranker
from citerx.retrieve.retriever import Retriever
from citerx.verify.claims import ABSTAIN, verify


class CiteRx:
    def __init__(self, settings: Settings | None = None, persist: bool = True):
        self.s = settings or get_settings()
        self.embedder = get_embedder(self.s.embedder, self.s.embed_model)
        self.index = DocumentIndex(self.embedder, self.s.index_dir if persist else None)
        self.reranker = get_reranker(self.s.reranker, self.s.rerank_model)
        self.retriever = Retriever(self.index, self.reranker, self.s)
        self.last_error: str | None = None

    # ---------- ingestion ----------
    def ingest(self, path: str | Path) -> list[Chunk]:
        chunks = chunk_file(path, self.s.max_chunk_chars, self.s.min_chunk_chars)
        self.index.add(chunks)
        return chunks

    def remove(self, source: str):
        self.index.remove({c.doc_id for c in self.index.chunks if c.source == source})

    # ---------- answering ----------
    def ask(self, question: str, use_llm: bool | None = None) -> Answer:
        sources = self.retriever.retrieve(question)
        use_llm = self.s.llm_enabled if use_llm is None else use_llm
        self.last_error = None

        if not sources:
            return Answer(question, NO_ANSWER, [], [], abstained=True, mode="extractive")

        text, mode = None, "extractive"
        if use_llm:
            try:
                text = chat(build_messages(question, sources), self.s.llm_base_url, self.s.llm_model, self.s.llm_api_key, self.s.llm_timeout)
                mode = "llm"
            except LLMError as e:
                self.last_error = f"{e} — fell back to extractive answer."
        if not text:
            text = extractive_answer(question, sources)

        abstained = text.strip() == NO_ANSWER or (len(text) < 200 and bool(ABSTAIN.search(text)))
        claims = [] if abstained else verify(text, sources, self.s.supported_threshold, self.s.weak_threshold)
        return Answer(question, text, sources, claims, abstained, mode)

    def describe(self) -> dict:
        return {
            "embedder": self.embedder.name,
            "reranker": self.reranker.name,
            "llm": f"{self.s.llm_model} @ {self.s.llm_base_url}" if self.s.llm_enabled else "off (extractive mode)",
            "documents": len(self.index.documents),
            "chunks": len(self.index.chunks),
        }
