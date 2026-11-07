"""Post-generation checks: every claim must be traceable to a retrieved source.

For each sentence in the draft answer:
  * it must cite at least one source [S#] that was actually provided   -> else "uncited"
  * each cited id must exist                                           -> else "invalid_citation"
  * every number in the sentence must appear in a cited source,
    unless the sentence shows a calculation                            -> else "number_not_in_source"

The answer is returned with status:
  answered    every sentence passed
  partial     some sentences failed; they are removed and listed in `removed`
  abstained   the model said INSUFFICIENT_EVIDENCE, or nothing survived verification
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .answer import ABSTAIN
from .index import Hit

CITE = re.compile(r"\[S(\d+)\]")
NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
CALC = re.compile(r"[=÷×*/]|\b(divided by|minus|plus|times|calculated|ratio of)\b", re.IGNORECASE)


def _norm_num(s: str) -> str:
    s = s.replace("$", "").replace(",", "").rstrip("%")
    return s[:-2] if s.endswith(".0") else s


def _numbers(text: str) -> set[str]:
    return {_norm_num(m) for m in NUM.findall(text) if len(_norm_num(m)) > 0}


@dataclass
class Verified:
    status: str
    text: str
    citations: list[dict] = field(default_factory=list)
    removed: list[dict] = field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\]])\s+(?=[A-Z(\"$])|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def verify(draft_text: str, hits: list[Hit]) -> Verified:
    if not draft_text or draft_text.strip().startswith(ABSTAIN):
        return Verified("abstained", "The documents provided do not contain enough information to answer this.")
    kept, removed, used = [], [], set()
    for sent in split_sentences(draft_text):
        ids = [int(x) for x in CITE.findall(sent)]
        if not ids:
            # a short heading or connective without a claim is allowed through
            if len(sent.split()) <= 4 and not NUM.search(sent):
                kept.append(sent)
                continue
            removed.append({"sentence": sent, "reason": "uncited"})
            continue
        if any(i < 1 or i > len(hits) for i in ids):
            removed.append({"sentence": sent, "reason": "invalid_citation"})
            continue
        claim = CITE.sub("", sent)
        source_nums = set().union(*(_numbers(hits[i - 1].chunk.text) for i in ids))
        missing = [n for n in _numbers(claim) if n not in source_nums and not re.fullmatch(r"20\d\d|19\d\d", n)]
        if missing and not CALC.search(claim):
            removed.append({"sentence": sent, "reason": "number_not_in_source", "numbers": missing})
            continue
        kept.append(sent)
        used.update(ids)
    citations = [{"id": f"S{i}", "doc": hits[i - 1].chunk.doc, "page": hits[i - 1].chunk.page + 1,
                  "chunk_id": hits[i - 1].chunk.chunk_id, "excerpt": hits[i - 1].chunk.text[:300]}
                 for i in sorted(used)]
    if not citations:
        return Verified("abstained", "No statement in the draft answer could be traced to a source.", [], removed)
    return Verified("answered" if not removed else "partial", "\n".join(kept), citations, removed)
