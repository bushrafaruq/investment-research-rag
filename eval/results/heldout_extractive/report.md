# Evaluation report: heldout split, `extractive` answerer

Corpus: 5 documents, 2145 chunks (2145 from company filings).

## 1. Retrieval (15 FinanceBench questions, page-level)

| Method | hit@1 | hit@3 | hit@6 | MRR |
|---|---|---|---|---|
| bm25 | 0.0 | 0.067 | 0.067 | 0.033 |
| bm25 + company filter | 0.0 | 0.2 | 0.267 | 0.106 |
| bm25 + company filter + query rewrite | 0.067 | 0.267 | 0.4 | 0.18 |
| tfidf | 0.0 | 0.133 | 0.133 | 0.067 |
| tfidf + company filter | 0.067 | 0.2 | 0.267 | 0.133 |
| tfidf + company filter + query rewrite | 0.0 | 0.267 | 0.333 | 0.122 |
| dense | 0.0 | 0.0 | 0.067 | 0.017 |
| dense + company filter | 0.067 | 0.2 | 0.333 | 0.139 |
| dense + company filter + query rewrite | 0.067 | 0.2 | 0.333 | 0.156 |
| lexical | 0.0 | 0.067 | 0.067 | 0.033 |
| lexical + company filter | 0.0 | 0.133 | 0.333 | 0.11 |
| lexical + company filter + query rewrite | 0.067 | 0.267 | 0.4 | 0.183 |
| hybrid | 0.0 | 0.067 | 0.133 | 0.044 |
| hybrid + company filter | 0.0 | 0.267 | 0.4 | 0.15 |
| hybrid + company filter + query rewrite | 0.0 | 0.2 | 0.4 | 0.127 |

## 2. Answers (15 FinanceBench questions)

| Metric | Value |
|---|---|
| Answered (all sentences verified) | 0.6 |
| Partial (some sentences removed by verification) | 0.0 |
| Abstained (false abstention on answerable questions) | 0.4 |
| Key-figure recall (gold number appears in answer; 15 questions) | 0.2 |
| Sentences removed by verification | 0 |
| Kept sentences without a citation | 0 |
