"""Turn source documents into page-aware chunks.

Every chunk keeps the document name and the page it came from, so every answer can cite
"document, page". Supported inputs: PDF, TXT, MD, HTML.

    python -m rag.cli ingest data/filings            # writes index/chunks.jsonl
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

CHUNK_WORDS = 350
OVERLAP_WORDS = 70


@dataclass
class Chunk:
    chunk_id: str
    doc: str          # file stem, e.g. AMD_2022_10K
    company: str      # from the file name before the first underscore, e.g. AMD
    page: int         # 0-based page index (as in the PDF)
    text: str
    quarantined: bool = False
    flags: tuple[str, ...] = ()


def read_pages(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in (".html", ".htm"):
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", text))
    # treat form feeds or "---page---" markers as page breaks; otherwise one page
    return re.split(r"\f|\n-{3,}\s*page\s*-{3,}\n", text, flags=re.IGNORECASE)


def clean(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"^\s*Table of Contents\s*$", " ", text, flags=re.MULTILINE)
    return re.sub(r"[ \t]+", " ", text).strip()


def chunk_page(doc: str, company: str, page: int, text: str) -> list[Chunk]:
    words = clean(text).split()
    if not words:
        return []
    out, step = [], CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, max(len(words) - OVERLAP_WORDS, 1), step):
        piece = " ".join(words[start:start + CHUNK_WORDS])
        cid = hashlib.sha1(f"{doc}|{page}|{start}".encode()).hexdigest()[:10]
        out.append(Chunk(cid, doc, company, page, piece))
    return out


def ingest(paths: list[Path], guard=None) -> list[Chunk]:
    """guard: optional callable(text) -> list of flag names; flagged chunks are quarantined."""
    chunks: list[Chunk] = []
    for path in sorted(paths):
        doc = path.stem
        company = doc.split("_")[0].upper()
        for i, page_text in enumerate(read_pages(path)):
            for c in chunk_page(doc, company, i, page_text):
                if guard:
                    flags = guard(c.text)
                    if flags:
                        c.quarantined, c.flags = True, tuple(flags)
                chunks.append(c)
    return chunks


def collect(source: str | Path) -> list[Path]:
    source = Path(source)
    exts = {".pdf", ".txt", ".md", ".html", ".htm"}
    if source.is_file():
        return [source]
    return [p for p in source.rglob("*") if p.suffix.lower() in exts and not p.name.startswith("README")]


def save(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load(path: str | Path) -> list[Chunk]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        d["flags"] = tuple(d.get("flags", ()))
        out.append(Chunk(**d))
    return out
