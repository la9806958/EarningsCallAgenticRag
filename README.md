## Context‑Enriched Earnings‑Call Dataset ##
*(companion to “Context‑Enriched Agent RAG: Cooperative LLM Retrieval for Predicting Post‑Earnings Price Shocks”)*

| Path | Contents | Rows × Cols† |
|------|----------|--------------|
| `maec_transcripts.csv` | **Text** calls from the MAEC benchmark (no return‑filter) | 2 725 × ~10 |
| `merged_data_nasdaq.csv` | NASDAQ calls with > 5 % return filter | 1 772 × ~14 |
| `merged_data_nyse.csv` | NYSE calls with > 5 % return filter | 1 386 × ~14 |
| `financial_statements/` | One CSV per ticker × statement type, **quarterly**, contains income statement, balance sheet, cash flow statement for each unique ticker | 8–60 rows each |

† Column counts are approximate because some files include additional derived features.
