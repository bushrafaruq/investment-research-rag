"""Command line.

    python -m rag.cli ingest data/filings [--out index/chunks.jsonl] [--embedder lsa|st:<model>|openai:<model>]
    python -m rag.cli ask "What drove AMD's revenue change in FY2022?" [--answerer extractive|llm|auto]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from . import guard
from .ingest import collect, ingest, save


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("source", nargs="+")
    i.add_argument("--out", default="index/chunks.jsonl")
    i.add_argument("--embedder", default=None, help="lsa (default), st:<model> or openai:<model>; or set RAG_EMBEDDER")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--index", default="index/chunks.jsonl")
    a.add_argument("--answerer", default="auto", choices=["auto", "llm", "extractive"])
    a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "ingest":
        paths = [p for s in args.source for p in collect(s)]
        chunks = ingest(paths, guard=guard.scan)
        save(chunks, args.out)
        from .index import HybridIndex
        from .pipeline import vector_cache_dir
        idx = HybridIndex(chunks, args.embedder)
        if not idx.vectors.embedder.needs_fit:
            idx.vectors.save(vector_cache_dir(args.out))      # model embeddings are reused at query time
        print(f"Vector index: {idx.vectors.embedder.name}, {idx.vectors.vectors.shape[0]} vectors "
              f"x {idx.vectors.vectors.shape[1]} dimensions")
        q = [c for c in chunks if c.quarantined]
        print(f"{len(paths)} documents, {len(chunks)} chunks, {len(q)} quarantined -> {args.out}")
        for doc, n in Counter(c.doc for c in chunks).items():
            print(f"  {doc}: {n} chunks")
        for c in q:
            print(f"  QUARANTINED {c.doc} p.{c.page + 1}: {', '.join(c.flags)}")
    else:
        from .pipeline import Copilot, to_dict
        res = Copilot(args.index, args.answerer).ask(args.question)
        if args.json:
            print(json.dumps(to_dict(res), indent=2))
            return
        print(f"[{res.status}] ({res.answerer}, {res.latency_ms} ms)\n{res.answer}\n")
        for c in res.citations:
            print(f"  {c['id']}: {c['doc']} p.{c['page']}")
        for r in res.removed:
            print(f"  removed ({r['reason']}): {r['sentence'][:100]}")
        if res.flags:
            print("  flags:", res.flags)


if __name__ == "__main__":
    main()
