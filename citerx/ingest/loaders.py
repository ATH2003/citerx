"""Read PDF, text and Markdown files into (page_number, text) pairs."""

from __future__ import annotations

from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}


def load_pages(path: str | Path) -> list[tuple[int, str]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}'. Use one of: {', '.join(sorted(SUPPORTED))}")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        if not any(text.strip() for _, text in pages):
            raise ValueError(f"No extractable text in {path.name} (scanned PDF? run OCR first).")
        return pages

    text = path.read_text(encoding="utf-8", errors="replace")
    # Form feeds mark page breaks in plain-text exports.
    return [(i + 1, page) for i, page in enumerate(text.split("\f"))]
