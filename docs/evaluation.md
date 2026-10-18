# Evaluation Framework & Benchmark Methodology

## 1. Overview and Core Objectives

The Private Legal Triage evaluation framework measures the correctness, grounding fidelity, safety, and performance of the local pipeline without relying on external evaluation APIs.

All evaluation runs locally using deterministic scripts (`evals/evaluate.py`) against synthetic contract datasets (`data/sample_contracts/`) and structured test suites.

---

## 2. Quantitative Evaluation Metrics

| Metric | Target | Formula / Definition | Enforcement |
| :--- | :--- | :--- | :--- |
| **Span Integrity Rate** | **100%** | $\frac{\text{Chunks where } \text{doc}[s:e] == \text{chunk.text}}{\text{Total Chunks}}$ | Deterministic test assertion |
| **Citation Verbatim Accuracy** | **100%** | $\frac{\text{Citations with exact verbatim text}}{\text{Total Citations}}$ | `citation_validator.py` |
| **Unsupported Claim Rate** | **0.0%** | $\frac{\text{Explicit claims without verified evidence}}{\text{Total Explicit Claims}}$ | `claim_validator.py` (Rule 1) |
| **Abstention Accuracy** | **> 98%** | Correct abstention when question asks about unstated terms | Evaluation benchmark suite |
| **Definitive Illegality Rate** | **0.0%** | Findings claiming definitive illegality / voidness | `response_guard.py` (Rule 4) |
| **Peak Memory Budget** | **< 6.0 GB** | Max resident RAM during local Gemma inference | `psutil` resource monitor |
| **BM25 Retrieval Latency** | **< 50 ms** | BM25 search & rank duration across document chunks | In-memory ranker timer |

---

## 3. Evaluation Datasets

1. **`evals/questions.jsonl`**:
   - Curated test queries spanning standard rental questions (rent, deposit, notice, pets, maintenance, entry rights).
   - Adversarial queries (prompt injection attempts, requests for definitive legal judgments).
   - Unanswerable queries (asking about terms not present in the contract).
2. **`evals/expected_evidence.jsonl`**:
   - Ground truth citations (expected page number, clause number, key quote substrings).
   - Expected status (`sufficient_evidence`, `insufficient_evidence`, `escalation_required`).
3. **Synthetic Rental Contracts (`data/sample_contracts/`)**:
   - Contract A: Standard complete tenancy agreement.
   - Contract B: Sparse agreement with missing clauses (no deposit clause, no notice period).
   - Contract C: Adversarial agreement with embedded prompt injection and contradictory clauses.

---

## 4. Running Local Evaluations

```bash
python evals/evaluate.py --contracts data/sample_contracts --questions evals/questions.jsonl
```
