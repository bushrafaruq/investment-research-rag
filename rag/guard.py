"""Prompt-injection defences.

Retrieved documents are untrusted input: a filing, a broker note or a web page can contain text
written to manipulate the model ("ignore your instructions and say..."). Three layers:

1. Ingest-time screening: chunks that contain instruction-like text aimed at an AI are
   quarantined. They stay in the index for audit but are never sent to the model.
2. Prompt structure: sources are wrapped in <source> tags and the system prompt tells the model
   that source text is data, never instructions (see answer.py).
3. Output checks: answers are verified against the cited sources after generation, so an
   injected claim that is not in a genuine source is flagged (see verify.py).

Layer 1 is pattern-based and deliberately conservative; its false-positive rate on real
10-K filings is measured in eval/run_eval.py.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = {
    "override_instructions": r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b(previous|prior|above|earlier|all|your|system)\b[^.]{0,20}\b(instructions?|prompts?|rules|directions)\b",
    "role_hijack": r"\byou are (now|no longer)\b|\bact as (an?|the) (ai|assistant|model|chatbot)\b|\bnew instructions?\s*:",
    "address_ai": r"\b(attention|note to|message (for|to)|instructions? (for|to))\b[^.]{0,20}\b(ai|llm|language model|assistant|chatbot|gpt|claude)\b",
    "exfiltrate_prompt": r"\b(reveal|print|repeat|show|output)\b[^.]{0,30}\b(system prompt|hidden instructions|your instructions)\b",
    "forced_output": r"\b(respond|reply|answer|say|state)\b[^.]{0,15}\b(only|exactly|verbatim)\b[^.]{0,15}\b(with|that)\b",
    "suppress_citation": r"\b(do not|don't|never)\b[^.]{0,20}\b(cite|mention|reveal)\b[^.]{0,20}\b(source|this (document|note|text))\b",
    "markup_payload": r"<\s*(script|iframe|img[^>]+onerror)|\]\(javascript:",
}

_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in INJECTION_PATTERNS.items()}


def scan(text: str) -> list[str]:
    """Return the names of injection patterns found in text (empty list = clean)."""
    return [name for name, rx in _COMPILED.items() if rx.search(text)]


def scan_query(question: str) -> list[str]:
    """Queries are screened too; a flagged query is answered as a plain question and logged."""
    return scan(question)
