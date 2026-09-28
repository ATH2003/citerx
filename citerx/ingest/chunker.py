"""Structure-aware chunking for clinical documents.

Rules, in order of priority:
1. Chunks never cross a section heading, and every chunk records its section title.
2. A safety block (WARNING, CAUTION, CONTRAINDICATED, BLACK BOX ...) is fused with the
   block that follows it, so a warning can never be retrieved without what it applies to,
   and vice versa. A fused unit is never split, even if it exceeds the size target.
3. Within those constraints, consecutive blocks are packed up to `max_chars`; tiny
   trailing chunks are merged back so retrieval isn't polluted by fragments.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from citerx.models import Chunk
from citerx.text import sentences

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^.:]{2,80})$")
_CAPS_HEADING = re.compile(r"^\s*([A-Z][A-Z0-9 ,:&/()\-]{3,70}?)\s*:?\s*$")
_TITLE_HEADING = re.compile(r"^\s*((?:[A-Z][a-z]+(?:-[A-Z]?[a-z]+)*)(?:\s+(?:[A-Z][a-z]+(?:-[A-Z]?[a-z]+)*|and|of|&|to|for|on))*)\s*:?\s*$")
_SAFETY = re.compile(
    r"^\s*(?:[*_>#\-\s]*)(WARNING|CAUTION|CONTRAINDICAT\w*|BLACK[- ]BOX|DANGER|ALERT|DO NOT|AVOID)\b",
    re.IGNORECASE,
)


@dataclass
class _Block:
    text: str
    page: int
    safety: bool


def _heading(line: str) -> str | None:
    if m := _MD_HEADING.match(line):
        return m.group(1).strip()
    if m := _NUMBERED_HEADING.match(line):
        return f"{m.group(1)} {m.group(2).strip()}"
    if (m := _CAPS_HEADING.match(line)) and not _SAFETY.match(line) and len(line.split()) <= 8:
        return m.group(1).strip().title()
    if (m := _TITLE_HEADING.match(line)) and len(line.split()) <= 5:
        return m.group(1).strip()
    return None


def _sections(pages: list[tuple[int, str]]) -> list[tuple[str, list[_Block]]]:
    """Walk pages line by line, grouping paragraph blocks under their heading."""
    sections: list[tuple[str, list[_Block]]] = [("", [])]
    buf: list[str] = []
    buf_page = 1

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            sections[-1][1].append(_Block(text, buf_page, bool(_SAFETY.match(text))))
        buf = []

    for page_no, page_text in pages:
        for line in page_text.splitlines():
            if not line.strip():
                flush()
                continue
            title = _heading(line)
            if title:
                flush()
                sections.append((title, []))
                continue
            if not buf:
                buf_page = page_no
            # A bullet or safety line starts a new block even without a blank line.
            if buf and (_SAFETY.match(line) or re.match(r"^\s*[-*•]\s", line) and not re.match(r"^\s*[-*•]\s", buf[-1])):
                flush()
                buf_page = page_no
            buf.append(line.rstrip())
        flush()
    return [(t, b) for t, b in sections if b]


def _units(blocks: list[_Block]) -> list[_Block]:
    """Fuse each safety block with its neighbour so they travel together."""
    units: list[_Block] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.safety and i + 1 < len(blocks):
            nxt = blocks[i + 1]
            units.append(_Block(f"{block.text}\n{nxt.text}", block.page, True))
            i += 2
            continue
        if block.safety and units:  # trailing warning: attach to what came before
            prev = units.pop()
            units.append(_Block(f"{prev.text}\n{block.text}", prev.page, True))
        else:
            units.append(block)
        i += 1
    return units


def _split_long(unit: _Block, max_chars: int) -> list[_Block]:
    if unit.safety or len(unit.text) <= max_chars:
        return [unit]
    out, cur = [], ""
    for sent in sentences(unit.text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            out.append(_Block(cur, unit.page, False))
            cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        out.append(_Block(cur, unit.page, False))
    return out


def chunk_pages(pages: list[tuple[int, str]], source: str, max_chars: int = 1100, min_chars: int = 200) -> list[Chunk]:
    doc_id = hashlib.sha1(source.encode()).hexdigest()[:10]
    chunks: list[Chunk] = []

    for section, blocks in _sections(pages):
        packed: list[_Block] = []
        for unit in (piece for u in _units(blocks) for piece in _split_long(u, max_chars)):
            last = packed[-1] if packed else None
            if last and len(last.text) + len(unit.text) + 1 <= max_chars:
                packed[-1] = _Block(f"{last.text}\n{unit.text}", last.page, last.safety or unit.safety)
            else:
                packed.append(unit)
        # Merge a runt tail into its predecessor.
        if len(packed) > 1 and len(packed[-1].text) < min_chars:
            tail = packed.pop()
            packed[-1] = _Block(f"{packed[-1].text}\n{tail.text}", packed[-1].page, packed[-1].safety or tail.safety)

        for block in packed:
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{idx:04d}",
                    doc_id=doc_id,
                    text=block.text,
                    source=source,
                    page=block.page,
                    section=section,
                    position=idx,
                    has_warning=block.safety,
                )
            )
    return chunks


def chunk_file(path: str | Path, max_chars: int = 1100, min_chars: int = 200) -> list[Chunk]:
    from citerx.ingest.loaders import load_pages

    path = Path(path)
    return chunk_pages(load_pages(path), path.name, max_chars, min_chars)
