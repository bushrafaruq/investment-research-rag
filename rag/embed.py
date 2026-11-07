"""Embeddings and the vector index.

Embedders turn text into dense, L2-normalised vectors; the VectorIndex stores one vector per
chunk and answers nearest-neighbour (cosine) queries. Three embedders share one interface:

    lsa                         Latent Semantic Analysis: TF-IDF -> truncated SVD (256 dimensions).
                                Fitted on the corpus itself, offline, no download. The default.
    st:<model>                  sentence-transformers model, e.g. st:sentence-transformers/all-MiniLM-L6-v2
                                (pip install sentence-transformers; downloads the model once)
    openai:<model>              OpenAI embeddings API, e.g. openai:text-embedding-3-small (OPENAI_API_KEY)

Choose with the RAG_EMBEDDER environment variable or --embedder on the CLI.

Vectors from model-based embedders are expensive to compute, so `rag.cli ingest` saves them next to
the chunks (index/<name>_vectors/vectors.npy + vectors.json) and they are reused when the chunk set and embedder match.
LSA is refitted at load time (a few seconds), because its vectors depend on the fitted corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_EMBEDDER = "lsa"


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder(Protocol):
    name: str
    needs_fit: bool

    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    """TF-IDF followed by truncated SVD: a classic dense embedding that needs no external model."""

    needs_fit = True

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim, self.seed = dim, seed
        self.name = f"lsa-{dim}"
        self._tfidf = self._svd = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from .index import tokenize
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                      sublinear_tf=True, min_df=2 if len(texts) >= 50 else 1)
        x = self._tfidf.fit_transform(texts)
        dim = min(self.dim, x.shape[1] - 1, x.shape[0] - 1)
        if dim < 2:
            self._svd = None         # corpus too small for SVD: use the TF-IDF vectors directly
            return
        self._svd = TruncatedSVD(n_components=dim, random_state=self.seed)
        self._svd.fit(x)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._tfidf is None:
            raise RuntimeError("LSAEmbedder.fit must be called before encode")
        x = self._tfidf.transform(texts)
        return _normalise(self._svd.transform(x) if self._svd is not None else x.toarray())


class SentenceTransformerEmbedder:
    needs_fit = False

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.name = f"st:{model}"

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                            show_progress_bar=False))


class OpenAIEmbedder:
    needs_fit = False

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client, self.model, self.name = OpenAI(), model, f"openai:{model}"

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            r = self.client.embeddings.create(model=self.model, input=texts[i:i + 256])
            out += [d.embedding for d in r.data]
        return _normalise(np.array(out))


def get_embedder(spec: str | None = None) -> Embedder:
    spec = spec or os.getenv("RAG_EMBEDDER", DEFAULT_EMBEDDER)
    kind, _, rest = spec.partition(":")
    if kind == "lsa":
        return LSAEmbedder(int(rest) if rest else 256)
    if kind == "st":
        return SentenceTransformerEmbedder(rest or "sentence-transformers/all-MiniLM-L6-v2")
    if kind == "openai":
        return OpenAIEmbedder(rest or "text-embedding-3-small")
    raise ValueError(f"unknown embedder {spec!r} (use lsa, st:<model> or openai:<model>)")


def corpus_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


class VectorIndex:
    """One L2-normalised vector per chunk; exact cosine search with an optional filter mask.

    Exact search over a few thousand chunks takes well under a millisecond with numpy. For
    millions of chunks this class is the place to swap in an approximate index (FAISS, HNSW or
    a vector database) without changing the rest of the pipeline.
    """

    def __init__(self, embedder: Embedder, ids: list[str], vectors: np.ndarray):
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors differ in length")
        self.embedder, self.ids, self.vectors = embedder, ids, vectors

    @classmethod
    def build(cls, embedder: Embedder, ids: list[str], texts: list[str],
              cache_dir: str | Path | None = None) -> VectorIndex:
        if embedder.needs_fit:
            embedder.fit(texts)
        cached = cls.load_vectors(cache_dir, embedder.name, ids) if cache_dir and not embedder.needs_fit else None
        vectors = cached if cached is not None else (embedder.encode(texts) if texts else np.zeros((0, 1), np.float32))
        return cls(embedder, ids, vectors)

    def search(self, query: str, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, cosine scores) of the k nearest chunks, best first."""
        if len(self.ids) == 0:
            return np.array([], dtype=int), np.array([])
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argsort(-scores, kind="stable")[:k]
        return top, scores[top]

    def scores(self, query: str) -> np.ndarray:
        return self.vectors @ self.embedder.encode([query])[0] if len(self.ids) else np.array([])

    # -- persistence -----------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "vectors.json").write_text(json.dumps({
            "embedder": self.embedder.name, "dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
            "count": len(self.ids), "corpus_hash": corpus_hash(self.ids), "ids": self.ids}), encoding="utf-8")

    @staticmethod
    def load_vectors(directory: str | Path, embedder_name: str, ids: list[str]) -> np.ndarray | None:
        d = Path(directory)
        meta_path, vec_path = d / "vectors.json", d / "vectors.npy"
        if not (meta_path.exists() and vec_path.exists()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedder") != embedder_name or meta.get("corpus_hash") != corpus_hash(ids):
            return None              # stale cache: different model or different chunks
        return np.load(vec_path)
