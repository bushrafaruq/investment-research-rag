"""Hybrid retrieval over three signals, merged with reciprocal rank fusion (RRF):

    bm25    exact terms: tickers, line items, defined terms
    tfidf   sparse vector cosine with word bigrams
    dense   embedding vectors in a VectorIndex (rag/embed.py): related wording without shared words

with filters for company and source type. The default embedder (LSA) needs no model download,
so the index can be rebuilt anywhere; sentence-transformers or OpenAI embeddings plug in via
RAG_EMBEDDER. See docs/DESIGN.md for the retrieval ablation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .embed import Embedder, VectorIndex, get_embedder
from .ingest import Chunk

# weights of (bm25, tfidf, dense) in the rank fusion
MODES = {
    "bm25": (1, 0, 0), "tfidf": (0, 1, 0), "dense": (0, 0, 1),
    "lexical": (1, 1, 0), "bm25+dense": (1, 0, 1), "hybrid": (1, 1, 1),
}
DEFAULT_MODE = "hybrid"

TOKEN = re.compile(r"[a-z0-9][a-z0-9.&'-]*[a-z0-9]|[a-z0-9]")
STOP = set("""a an the and or of to in on for by with from as at is are was were be been this that these those
it its our we us their they which who whom what when where how why than then there here also such any all
each other into over under per not no do does did has have had will would can could may might shall should""".split())

COMPANY_ALIASES = {
    "AMD": ["amd", "advanced micro devices"],
    "AMERICANEXPRESS": ["american express", "amex", "americanexpress"],
    "BOEING": ["boeing"],
    "PEPSICO": ["pepsico", "pepsi"],
    "3M": ["3m"],
    "CVSHEALTH": ["cvs health", "cvs"],
    "JOHNSON": ["johnson & johnson", "johnson and johnson", "j&j", "jnj"],
    "BESTBUY": ["best buy", "bestbuy"],
    "AES": ["aes"],
}


PRIMARY = re.compile(r"_(10K|10Q|8K|EARNINGS)\b", re.IGNORECASE)


def is_primary(doc: str) -> bool:
    """Company filings and earnings releases are primary sources; anything else (broker notes,
    blogs, web pages) is secondary and only searched when explicitly allowed."""
    return bool(PRIMARY.search(doc))


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float        # fused score
    bm25: float
    cosine: float       # TF-IDF cosine
    dense: float        # embedding cosine (0 when the dense signal is not used)
    rank: int


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out[i] = s
        return out


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: str | Embedder | None = None,
                 vector_cache: str | None = None):
        self.all_chunks = chunks
        # quarantined chunks are kept for audit but never retrieved or embedded
        self.chunks = [c for c in chunks if not c.quarantined]
        texts = [c.text for c in self.chunks]
        toks = [tokenize(t) for t in texts]
        self.bm25 = BM25(toks)
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                     ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.matrix = self.tfidf.fit_transform(texts) if self.chunks else None
        emb = embedder if embedder is not None and not isinstance(embedder, str) else get_embedder(embedder)
        self.vectors = VectorIndex.build(emb, [c.chunk_id for c in self.chunks], texts, vector_cache)
        self.companies = sorted({c.company for c in self.chunks})

    def detect_companies(self, question: str) -> list[str]:
        q = question.lower()
        found = [co for co, names in COMPANY_ALIASES.items()
                 if co in self.companies and any(re.search(rf"\b{re.escape(n)}\b", q) for n in names)]
        # documents not in the alias table: match on the company code from the file name
        found += [co for co in self.companies if co not in COMPANY_ALIASES and re.search(rf"\b{co.lower()}\b", q)]
        return found

    def search(self, question: str, k: int = 5, companies: list[str] | None = None,
               primary_only: bool = True, mode: str = DEFAULT_MODE, rrf_k: int = 60) -> list[Hit]:
        """mode: one of MODES; the single-signal modes exist for evaluation."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if not self.chunks:
            return []
        w_bm, w_cos, w_dense = MODES[mode]
        q = tokenize(question)
        bm = self.bm25.scores(q)
        cos = (self.matrix @ self.tfidf.transform([question]).T).toarray().ravel()
        dense = self.vectors.scores(question) if w_dense else np.zeros(len(self.chunks))
        mask = np.ones(len(self.chunks), dtype=bool)
        if companies:
            mask &= np.array([c.company in companies for c in self.chunks])
        if primary_only:
            mask &= np.array([is_primary(c.doc) for c in self.chunks])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        def ranks(scores):
            return {i: r for r, i in enumerate(idx[np.argsort(-scores[idx], kind="stable")])}

        bm_rank, cos_rank, dense_rank = ranks(bm), ranks(cos), ranks(dense)
        fused = {i: w_bm / (rrf_k + bm_rank[i]) + w_cos / (rrf_k + cos_rank[i]) + w_dense / (rrf_k + dense_rank[i])
                 for i in idx}
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.chunks[i], fused[i], float(bm[i]), float(cos[i]), float(dense[i]), r)
                for r, i in enumerate(top)]
