import pytest

from citerx.ingest.chunker import chunk_pages
from citerx.models import Chunk, ScoredChunk
from citerx.pipeline import CiteRx
from citerx.config import ROOT, Settings


def make_sources(*texts: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(Chunk(f"c{i}", "d", t, "doc.txt", 1, "Section", i), 1.0)
        for i, t in enumerate(texts)
    ]


@pytest.fixture(scope="session")
def rx(tmp_path_factory) -> CiteRx:
    settings = Settings()
    settings.index_dir = tmp_path_factory.mktemp("index")
    engine = CiteRx(settings)
    for f in sorted((ROOT / "samples").iterdir()):
        engine.ingest(f)
    return engine


@pytest.fixture
def chunk():
    def _chunk(text: str, **kw):
        return chunk_pages([(1, text)], "test.txt", **kw)
    return _chunk
