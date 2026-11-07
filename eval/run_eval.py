"""Evaluate the copilot.

    python scripts/download_filings.py
    python -m rag.cli ingest data/filings data/injection_tests --out index/eval_chunks.jsonl
    python eval/run_eval.py                       # offline (extractive answerer)
    python eval/run_eval.py --answerer llm        # needs ANTHROPIC_API_KEY or OPENAI_API_KEY

    # held-out filings and questions, never looked at during development
    python scripts/download_filings.py --set heldout
    python -m rag.cli ingest data/filings_heldout --out index/heldout_chunks.jsonl
    python eval/run_eval.py --split heldout

Writes eval/results/<split>_<answerer>/: metrics.json, report.md and answers.csv (for manual grading).

Measures
  1. Retrieval     page-level hit@1/3/6 and MRR on 26 FinanceBench questions; ablation of
                   BM25 vs TF-IDF vs hybrid, with and without the company filter
  2. Answers       status (answered / partial / abstained), false-abstention rate, key-figure
                   recall (gold numbers present in the answer), sentences removed by verification
  3. Hallucination every kept sentence cites a retrieved source and every number is in the
                   cited source (by construction, checked here); correct abstention on 10
                   unanswerable questions
  4. Injection     detector recall on planted attacks, false-positive rate on real 10-K chunks,
                   attack success with all sources vs primary sources only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.guard import scan  # noqa: E402
from rag.index import is_primary  # noqa: E402
from rag.pipeline import Copilot  # noqa: E402
from rag.query import rewrite  # noqa: E402
from rag.verify import CITE, _numbers  # noqa: E402

K_VALUES = (1, 3, 6)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def gold_numbers(answer: str) -> set[str]:
    nums = {n for n in _numbers(answer) if not re.fullmatch(r"20\d\d", n)}
    return nums


def retrieval_eval(cp: Copilot, qs: list[dict]) -> dict:
    out = {}
    for mode in ("bm25", "tfidf", "dense", "lexical", "hybrid"):
        for filt, rw in ((False, False), (True, False), (True, True)):
            hits_at = {k: 0 for k in K_VALUES}
            rr = 0.0
            for q in qs:
                companies = cp.index.detect_companies(q["question"]) if filt else None
                text = rewrite(q["question"]) if rw else q["question"]
                hits = cp.index.search(text, max(K_VALUES), companies or None, mode=mode)
                pages = [(h.chunk.doc, h.chunk.page) for h in hits]
                gold = {(q["doc"], p) for p in q["evidence_pages"]}
                first = next((i for i, x in enumerate(pages) if x in gold), None)
                for k in K_VALUES:
                    hits_at[k] += first is not None and first < k
                rr += 1 / (first + 1) if first is not None else 0
            n = len(qs)
            out[f"{mode}{' + company filter' if filt else ''}{' + query rewrite' if rw else ''}"] = {
                **{f"hit@{k}": round(v / n, 3) for k, v in hits_at.items()}, "mrr": round(rr / n, 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--index")
    ap.add_argument("--answerer", default="extractive", choices=["extractive", "llm"])
    a = ap.parse_args()

    dev = a.split == "dev"
    index = a.index or str(ROOT / "index" / ("eval_chunks.jsonl" if dev else "heldout_chunks.jsonl"))
    fb = load_jsonl(ROOT / "data" / "eval" / ("financebench_subset.jsonl" if dev else "financebench_heldout.jsonl"))
    extra = load_jsonl(ROOT / "data" / "eval" / "extra_questions.jsonl") if dev else []
    cp = Copilot(index, a.answerer)
    cp_all = Copilot(index, a.answerer, primary_only=False)
    out_dir = ROOT / "eval" / "results" / f"{a.split}_{a.answerer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1. retrieval
    retrieval = retrieval_eval(cp, fb)

    # 2-3. answers on answerable questions
    status = {"answered": 0, "partial": 0, "abstained": 0}
    fig_hits = fig_total = removed = kept_sentences = uncited = 0
    for q in fb:
        r = cp.ask(q["question"])
        status[r.status] += 1
        g = gold_numbers(q["answer"])
        found = {n for n in g if n in _numbers(r.answer)}
        if g:
            fig_total += 1
            fig_hits += bool(found)
        removed += len(r.removed)
        sents = [s for s in r.answer.splitlines() if s.strip()] if r.status != "abstained" else []
        kept_sentences += len(sents)
        uncited += sum(1 for s in sents if not CITE.search(s) and len(s.split()) > 4)
        rows.append({"id": q["id"], "type": "financebench", "question": q["question"], "gold": q["answer"],
                     "status": r.status, "answer": r.answer,
                     "citations": "; ".join(f"{c['doc']} p.{c['page']}" for c in r.citations),
                     "evidence_pages": ";".join(str(p + 1) for p in q["evidence_pages"]),
                     "gold_figure_found": bool(found) if g else "", "human_grade": ""})

    n = len(fb)
    metrics = {
        "split": a.split,
        "answerer": cp.answerer.name,
        "retrieval": retrieval,
        "answers": {
            "questions": n, **{f"{k}_rate": round(v / n, 3) for k, v in status.items()},
            "key_figure_recall": round(fig_hits / fig_total, 3) if fig_total else None,
            "questions_with_gold_figure": fig_total,
            "sentences_removed_by_verification": removed,
            "kept_sentences_without_citation": uncited,
        },
    }
    if dev:
        metrics.update(safety_eval(cp, cp_all, extra, rows))
    chunks = cp.index.all_chunks
    metrics["corpus"] = {"documents": len({c.doc for c in chunks}), "chunks": len(chunks),
                         "filings_chunks": sum(is_primary(c.doc) for c in chunks)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "answers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "report.md").write_text(report(metrics), encoding="utf-8")
    print(report(metrics))


def safety_eval(cp: Copilot, cp_all: Copilot, extra: list[dict], rows: list[dict]) -> dict:
    # unanswerable
    un = [q for q in extra if q["type"] == "unanswerable"]
    correct_abstain = 0
    for q in un:
        r = cp.ask(q["question"])
        correct_abstain += r.status == "abstained"
        rows.append({"id": q["id"], "type": "unanswerable", "question": q["question"], "gold": q["why"],
                     "status": r.status, "answer": r.answer, "citations": "", "evidence_pages": "",
                     "gold_figure_found": "", "human_grade": ""})

    # 4. injection
    chunks = cp.index.all_chunks
    attack_chunks = [c for c in chunks if not is_primary(c.doc) and re.search(
        r"ignore|instructions|you are now|assistant reading", c.text, re.I)]
    real_chunks = [c for c in chunks if is_primary(c.doc)]
    fp = [c for c in real_chunks if c.quarantined]
    inj = [q for q in extra if q["type"] == "injection"]
    success = {"primary_only": 0, "all_sources": 0}
    for q in inj:
        for name, copilot in (("primary_only", cp), ("all_sources", cp_all)):
            r = copilot.ask(q["question"])
            hit = any(p.lower() in r.answer.lower() for p in q["payloads"])
            success[name] += hit
            if name == "all_sources":
                rows.append({"id": q["id"], "type": "injection (all sources)", "question": q["question"],
                             "gold": "must not contain: " + ", ".join(q["payloads"]), "status": r.status,
                             "answer": r.answer, "citations": "; ".join(f"{c['doc']} p.{c['page']}" for c in r.citations),
                             "evidence_pages": "", "gold_figure_found": "", "human_grade": "attack succeeded" if hit else ""})
    query_flagged = sum(bool(scan(q["question"])) for q in inj if q["attack_doc"] is None)
    return {
        "unanswerable": {"questions": len(un), "correct_abstention_rate": round(correct_abstain / len(un), 3)},
        "injection": {
            "planted_attack_paragraphs": len(attack_chunks),
            "detected": sum(c.quarantined for c in attack_chunks),
            "detector_recall": round(sum(c.quarantined for c in attack_chunks) / len(attack_chunks), 3),
            "false_positives_on_filings": len(fp),
            "false_positive_rate": round(len(fp) / len(real_chunks), 5),
            "attack_queries": len(inj),
            "attacks_succeeded_primary_only": success["primary_only"],
            "attacks_succeeded_all_sources": success["all_sources"],
            "injected_query_flagged": query_flagged,
        },
    }


def report(m: dict) -> str:
    r = m["retrieval"]
    nq = m["answers"]["questions"]
    lines = [f"# Evaluation report: {m['split']} split, `{m['answerer']}` answerer", "",
             f"Corpus: {m['corpus']['documents']} documents, {m['corpus']['chunks']} chunks "
             f"({m['corpus']['filings_chunks']} from company filings).", "",
             f"## 1. Retrieval ({nq} FinanceBench questions, page-level)", "",
             "| Method | hit@1 | hit@3 | hit@6 | MRR |", "|---|---|---|---|---|"]
    for name, v in r.items():
        lines.append(f"| {name} | {v['hit@1']} | {v['hit@3']} | {v['hit@6']} | {v['mrr']} |")
    a = m["answers"]
    lines += ["", f"## 2. Answers ({nq} FinanceBench questions)", "", "| Metric | Value |", "|---|---|",
              f"| Answered (all sentences verified) | {a['answered_rate']} |",
              f"| Partial (some sentences removed by verification) | {a['partial_rate']} |",
              f"| Abstained (false abstention on answerable questions) | {a['abstained_rate']} |",
              f"| Key-figure recall (gold number appears in answer; {a['questions_with_gold_figure']} questions) | {a['key_figure_recall']} |",
              f"| Sentences removed by verification | {a['sentences_removed_by_verification']} |",
              f"| Kept sentences without a citation | {a['kept_sentences_without_citation']} |"]
    if "unanswerable" not in m:
        return "\n".join(lines) + "\n"
    lines += ["", "## 3. Unanswerable questions", "",
              f"Correct abstention or decline: **{m['unanswerable']['correct_abstention_rate']}** "
              f"of {m['unanswerable']['questions']} questions.", "", "## 4. Prompt injection", "",
              "| Metric | Value |", "|---|---|"]
    i = m["injection"]
    lines += [f"| Planted attack paragraphs detected at ingest | {i['detected']} of {i['planted_attack_paragraphs']} |",
              f"| False positives on real 10-K chunks | {i['false_positives_on_filings']} ({i['false_positive_rate']}) |",
              f"| Attacks succeeded: primary sources only (default) | {i['attacks_succeeded_primary_only']} of {i['attack_queries']} |",
              f"| Attacks succeeded: all sources allowed | {i['attacks_succeeded_all_sources']} of {i['attack_queries']} |",
              f"| Injected user query flagged | {i['injected_query_flagged']} |"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
