"""The copilot: question -> scope -> retrieve -> answerability check -> answer -> verify."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import guard
from .answer import ExtractiveAnswerer, LLMAnswerer
from .embed import Embedder
from .index import COMPANY_ALIASES, HybridIndex
from .ingest import load
from .query import rewrite
from .verify import verify

KNOWN_UNCOVERED = ["nvidia", "intel", "apple", "microsoft", "coca-cola", "coca cola", "airbus", "visa", "mastercard"]
ADVICE = re.compile(r"\b(should i|should we|is it a good)\b.*\b(buy|sell|invest|hold)\b|\b(price target|"
                    r"(share|stock) price (will|be|next|at the end))\b|\bwill .*\b(stock|shares?) (go|rise|fall)\b", re.I)
PERSONAL = re.compile(r"\b(home address|personal (phone|email|address)|where does .* live)\b", re.I)


@dataclass
class Answer:
    question: str
    status: str                     # answered | partial | abstained
    answer: str
    citations: list[dict]
    removed: list[dict]
    companies: list[str]
    retrieved: list[dict]
    flags: list[str] = field(default_factory=list)
    answerer: str = ""
    latency_ms: int = 0


def vector_cache_dir(chunks_path: str | Path) -> Path:
    """Saved embedding vectors live next to the chunks: index/chunks.jsonl -> index/chunks_vectors/."""
    p = Path(chunks_path)
    return p.with_name(p.stem + "_vectors")


class Copilot:
    def __init__(self, chunks_path: str | Path = "index/chunks.jsonl", answerer: str = "auto", k: int = 6,
                 primary_only: bool = True, embedder: str | Embedder | None = None):
        self.index = HybridIndex(load(chunks_path), embedder, vector_cache_dir(chunks_path))
        self.k, self.primary_only = k, primary_only
        if answerer == "extractive":
            self.answerer = ExtractiveAnswerer()
        elif answerer == "llm":
            self.answerer = LLMAnswerer()
        else:
            try:
                self.answerer = LLMAnswerer()
            except RuntimeError:
                self.answerer = ExtractiveAnswerer()

    def coverage_problem(self, question: str, companies: list[str], hits) -> str | None:
        """Cheap, explainable checks before spending a model call."""
        q = question.lower()
        if ADVICE.search(question):
            return ("I can't give investment advice or price forecasts. I can summarise what the filings say, "
                    "for example about results, risks or guidance.")
        if PERSONAL.search(question):
            return "I can't help find personal information about individuals."
        if any(re.search(rf"\b{n}\b", q) for n in KNOWN_UNCOVERED):
            return "The question is about a company that is not in the document set."
        if not companies:
            if len(self.index.companies) > 1:
                return ("Please name the company. Documents are available for: "
                        + ", ".join(self.index.companies) + ".")
        years = set(re.findall(r"\b(?:fy\s?)?(20\d\d)\b", q))
        if years and hits:
            text = " ".join(h.chunk.text for h in hits) + " " + " ".join(h.chunk.doc for h in hits)
            missing = [y for y in years if y not in text]
            if missing:
                return f"The documents do not cover {', '.join(sorted(missing))}."
        if not hits:
            return "No relevant passages were found."
        return None

    def ask(self, question: str, k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        flags = [f"query_{f}" for f in guard.scan_query(question)]
        companies = self.index.detect_companies(question)
        hits = self.index.search(rewrite(question), k or self.k, companies or None, primary_only=self.primary_only)
        retrieved = [{"rank": h.rank + 1, "doc": h.chunk.doc, "page": h.chunk.page + 1, "score": round(h.score, 4),
                      "chunk_id": h.chunk.chunk_id} for h in hits]
        problem = self.coverage_problem(question, companies, hits)
        if problem:
            return Answer(question, "abstained", problem, [], [], companies, retrieved, flags,
                          getattr(self.answerer, "name", ""), int((time.perf_counter() - t0) * 1000))
        draft = self.answerer.answer(question, hits)
        v = verify(draft.text, hits)
        return Answer(question, v.status, v.text, v.citations, v.removed, companies, retrieved, flags,
                      draft.model, int((time.perf_counter() - t0) * 1000))

    def documents(self) -> list[dict]:
        docs = {}
        for c in self.index.all_chunks:
            d = docs.setdefault(c.doc, {"doc": c.doc, "company": c.company, "chunks": 0, "quarantined": 0, "pages": 0})
            d["chunks"] += 1
            d["quarantined"] += c.quarantined
            d["pages"] = max(d["pages"], c.page + 1)
        return list(docs.values())


def to_dict(a: Answer) -> dict:
    return asdict(a)


__all__ = ["COMPANY_ALIASES", "Answer", "Copilot", "to_dict"]
