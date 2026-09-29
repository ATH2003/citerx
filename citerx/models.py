"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    """A retrievable passage of a document, with provenance."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    page: int
    section: str
    position: int
    has_warning: bool = False

    def embed_text(self) -> str:
        """Text used for indexing: the section title adds context to short passages."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    signals: dict = field(default_factory=dict)  # e.g. {"bm25_rank": 2, "dense_rank": 1}


@dataclass
class ClaimCheck:
    """Verification result for one sentence of a generated answer."""

    sentence: str
    cited: list[int]
    status: str  # "supported" | "weak" | "unsupported" | "uncited"
    support: float
    evidence: str = ""
    missing_numbers: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[ScoredChunk]
    claims: list[ClaimCheck]
    abstained: bool
    mode: str  # "llm" | "extractive"

    @property
    def groundedness(self) -> float:
        """Share of claims fully supported by the sources they cite."""
        if not self.claims:
            return 1.0 if self.abstained else 0.0
        return sum(c.status == "supported" for c in self.claims) / len(self.claims)

    @property
    def flagged(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.status in {"unsupported", "uncited"}]
