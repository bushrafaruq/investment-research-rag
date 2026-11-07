"""HTTP API.

    uvicorn rag.api:app --port 8000
    curl -X POST localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "What drove AMD revenue change in FY2022?"}'
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import Copilot, to_dict

app = FastAPI(title="Investment Research RAG Copilot", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=6, ge=1, le=20)


@lru_cache(maxsize=1)
def copilot() -> Copilot:
    path = os.getenv("RAG_INDEX", "index/chunks.jsonl")
    if not os.path.exists(path):
        raise HTTPException(503, f"Index not found at {path}. Run: python -m rag.cli ingest data/filings")
    return Copilot(path, os.getenv("RAG_ANSWERER", "auto"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/documents")
def documents() -> list[dict]:
    return copilot().documents()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return to_dict(copilot().ask(req.question, req.k))
