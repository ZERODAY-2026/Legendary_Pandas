# Evaluation Framework & Benchmark Methodology

## 1. Objectives
The evaluation suite objectively assesses the accuracy, safety, grounding, and performance of the local legal triage pipeline without human intervention.

---

## 2. Benchmark Metrics

| Metric | Definition | Target |
|---|---|---|
| **Citation Accuracy** | Percentage of cited quotes that are verbatim substrings of the original document. | **100%** (zero tolerance for fabricated quotes) |
| **Span Precision** | Verification that `raw_text[span_start:span_end] == chunk.text`. | **100%** |
| **Abstention Quality** | Percentage of questions about unmentioned topics where the system correctly returns `"insufficient_evidence"`. | **> 95%** |
| **Unsupported Claim Rate** | Findings with non-null `explicit_text` lacking matching evidence quotes. | **0.0%** (enforced deterministically by schema) |
| **Definitive Legality Violations** | Instances where the model asserts definitive illegality without escalation. | **0.0%** |
| **Peak Memory Footprint** | RAM consumed during full ingestion, retrieval, and inference. | **< 6.0 GB** (fits commodity laptops) |
| **Inference Latency** | End-to-end query turnaround time on CPU/Metal/CUDA. | **< 8.0s** per query |

---

## 3. Evaluation Datasets
- **`evals/questions.jsonl`**: Curated test questions covering obligations, deadlines, penalties, contract silence, ambiguous clauses, and requests for definitive legal judgments.
- **`evals/expected_evidence.jsonl`**: Ground truth mapping containing expected source IDs, required verbatim keywords, expected finding categories, and escalation status.

---

## 4. Running the Benchmark
```bash
python evals/evaluate.py
```
Outputs a structured summary of metric scores, latency percentiles (p50, p95), and validation failure counts.
