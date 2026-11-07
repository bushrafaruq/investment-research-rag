# Evaluation data

* `financebench_subset.jsonl`: the 26 questions in the FinanceBench open-source sample that refer to
  the four filings used here (AMD, American Express, Boeing and PepsiCo FY2022 10-Ks), with the
  gold answer and the 0-based evidence page(s). Source: FinanceBench, Islam et al. (2023),
  https://github.com/patronus-ai/financebench. See that repository for licence terms; the data is
  used here for non-commercial evaluation with attribution.
* `extra_questions.jsonl`: written for this project. Questions the copilot should decline or
  abstain on (other companies, years not covered, investment advice, personal data, facts not in
  the filings) and prompt-injection attacks that target the test documents in
  `../injection_tests/`.
