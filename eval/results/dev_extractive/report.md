# Evaluation report: dev split, `extractive` answerer

Corpus: 7 documents, 1858 chunks (1849 from company filings).

## 1. Retrieval (26 FinanceBench questions, page-level)

| Method | hit@1 | hit@3 | hit@6 | MRR |
|---|---|---|---|---|
| bm25 | 0.077 | 0.154 | 0.231 | 0.133 |
| bm25 + company filter | 0.077 | 0.231 | 0.269 | 0.163 |
| bm25 + company filter + query rewrite | 0.269 | 0.462 | 0.654 | 0.4 |
| tfidf | 0.077 | 0.115 | 0.115 | 0.09 |
| tfidf + company filter | 0.077 | 0.154 | 0.231 | 0.122 |
| tfidf + company filter + query rewrite | 0.308 | 0.5 | 0.577 | 0.404 |
| dense | 0.038 | 0.154 | 0.269 | 0.113 |
| dense + company filter | 0.154 | 0.231 | 0.269 | 0.2 |
| dense + company filter + query rewrite | 0.346 | 0.423 | 0.577 | 0.418 |
| lexical | 0.077 | 0.154 | 0.192 | 0.117 |
| lexical + company filter | 0.077 | 0.231 | 0.269 | 0.144 |
| lexical + company filter + query rewrite | 0.269 | 0.538 | 0.654 | 0.407 |
| hybrid | 0.077 | 0.231 | 0.231 | 0.135 |
| hybrid + company filter | 0.115 | 0.231 | 0.308 | 0.183 |
| hybrid + company filter + query rewrite | 0.308 | 0.5 | 0.654 | 0.429 |

## 2. Answers (26 FinanceBench questions)

| Metric | Value |
|---|---|
| Answered (all sentences verified) | 0.846 |
| Partial (some sentences removed by verification) | 0.0 |
| Abstained (false abstention on answerable questions) | 0.154 |
| Key-figure recall (gold number appears in answer; 11 questions) | 0.273 |
| Sentences removed by verification | 0 |
| Kept sentences without a citation | 0 |

## 3. Unanswerable questions

Correct abstention or decline: **0.9** of 10 questions.

## 4. Prompt injection

| Metric | Value |
|---|---|
| Planted attack paragraphs detected at ingest | 3 of 4 |
| False positives on real 10-K chunks | 0 (0.0) |
| Attacks succeeded: primary sources only (default) | 0 of 4 |
| Attacks succeeded: all sources allowed | 0 of 4 |
| Injected user query flagged | 1 |
