"""Query rewriting for retrieval (the original question is still what the answerer sees).

1. Remove words that match almost every chunk of a single company's filing and add noise:
   the company's own name (already used as a filter) and fiscal-year markers.
2. Remove analyst boilerplate ("If gross margin is not a useful metric ..., state that ...").
3. Expand finance concepts into the line-item wording used in 10-K filings, e.g.
   "quick ratio" -> cash, short-term investments, accounts receivable, current liabilities.

The glossary was written while developing against the dev question set, so dev-set retrieval
scores are optimistic. eval/run_eval.py --split heldout measures it on filings and questions
that were not looked at during development.
"""

from __future__ import annotations

import re

from .index import COMPANY_ALIASES

GLOSSARY = {
    r"quick ratio": "cash and cash equivalents short-term investments accounts receivable total current liabilities",
    r"current ratio|liquidity": "total current assets total current liabilities liquidity capital resources",
    r"operating margin|operating profit": "operating income net revenue",
    r"gross margin|gross profit": "gross margin gross profit cost of sales net revenue",
    r"effective tax rate|tax rate": "effective tax rate income tax provision benefit",
    r"capex|capital expenditures?": "purchases of property plant and equipment capital spending",
    r"ebitda": "operating profit depreciation amortization",
    r"balance sheet|liabilit": "consolidated balance sheets total liabilities",
    r"cash flow": "net cash provided by used in operating investing financing activities",
    r"revenue change|drove revenue|revenue growth": "net revenue increased decreased primarily due",
    r"legal|lawsuit|litigation": "legal proceedings litigation lawsuits",
    r"customer concentration|major customers?": "customer accounted for percent of net revenue",
    r"primary customers?": "customers commercial airlines government",
    r"geograph|regions?|countries": "geographic segments regions international",
    r"products? and services|what .* sells?": "products segments business overview",
    r"cyclical": "cyclical economic conditions downturn",
    r"retain|retention|attrition": "retention attrition loyalty",
    r"restructuring": "restructuring and impairment charges",
    r"dividend": "dividends paid per share",
    r"debt securities|registered": "securities registered pursuant to section 12(b)",
    r"segments?": "segment net revenue operating income",
}

BOILERPLATE = [
    r"if [^.?]*? is not a useful metric[^.?]*[.?]",
    r"give a response to the question by relying on[^.?]*[.?]",
    r"answer in usd [a-z]+\.?|\(in usd [a-z]+\)",
    r"round your answer[^.?]*[.?]",
    r"please (state|explain)[^.?]*[.?]",
]


def rewrite(question: str) -> str:
    q = question.lower()
    for pat in BOILERPLATE:
        q = re.sub(pat, " ", q)
    for names in COMPANY_ALIASES.values():
        for n in names:
            q = re.sub(rf"\b{re.escape(n)}('s)?\b", " ", q)
    q = re.sub(r"\bfy\s?'?\d{2,4}\b|\bfiscal (year )?\d{4}\b|\bas of\b", " ", q)
    extra = [exp for pat, exp in GLOSSARY.items() if re.search(pat, question, re.IGNORECASE)]
    return re.sub(r"\s+", " ", " ".join([q, *extra])).strip()
