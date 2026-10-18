# Private Legal Triage

> A local-first, evidence-grounded legal document triage system for rental agreements (PDF and TXT).

Private Legal Triage is **not a general legal chatbot**. It ingests a residential tenancy agreement, extracts its text locally, identifies key clauses, answers user questions using verifiable verbatim evidence from the document, explains potential concerns, and explicitly reports when the document does not contain enough information.

---

## Hard Privacy & Architectural Constraints

- **100% Local Execution**: All processing (parsing, chunking, retrieval indexing, prompt generation, Gemma inference, verification, and temporary files) stays strictly on the user's device.
- **No Cloud Fallbacks**: No cloud LLM APIs, hosted vector databases, remote embeddings, cloud OCR, or telemetry beacons. If local inference fails, the system fails closed with a transparent error.
- **Local Loopback Only**: The FastAPI server strictly binds to `127.0.0.1` and rejects remote network access.
- **Content-Free Logs**: Logs never record document content, user questions, party names, or model answers.
- **Source Separation**: Document sources are tagged at ingestion into three immutable types (`contract`, `legislation`, `court_judgment`). External law is never conflated with the uploaded contract.
- **Untrusted Evidence**: Contract text is treated strictly as untrusted data within an isolated prompt envelope to prevent prompt injection.
- **Deterministic Verification**: Every finding and citation is validated against the original text before display. Questions requiring definitive legal judgments trigger mandatory escalation to a legal professional.

---

## Deterministic Rules Enforced on Every Finding

1. **Explicit Statement**: Must be supported by a real verbatim passage from the source document.
2. **Interpretation**: Must be explicitly labelled as an interpretation.
3. **Missing Clause**: Must not be described as one that exists in the document.
4. **Legal Concern**: Must never be a definitive finding of illegality or invalidity.
5. **Source Separation**: Legislation and court judgments must never be cited as terms of the uploaded contract.
6. **Insufficient Evidence**: If evidence is inadequate, the answer must state what could not be established.

---

## Tech Stack & Hardware Requirements

- **Backend**: Python 3.11, FastAPI, Pydantic v2, PyMuPDF (`fitz`), `rank-bm25`
- **Frontend**: React 18, TypeScript, Vite, Vanilla CSS + Tailwind-free design system, restrictive CSP
- **Model**: Gemma Edge variant (`gemma-2-2b-it` / `gemma-3n` Q4_K_M quantized GGUF)
- **Host Specs**: CPU execution supported; ~16 GB RAM; 2-4 GB memory budget for Gemma runtime.

---

## Repository Structure

```
private-legal-triage/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── api/
│   │   └── types/schemas.ts
│   └── package.json
├── backend/
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── api/
│       ├── ingestion/
│       ├── document/schemas.py
│       ├── retrieval/
│       ├── agent/
│       ├── legal_sources/
│       │   ├── registry.py
│       │   └── provenance.py
│       ├── safety/
│       └── storage/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── safety/
│   ├── privacy/
│   └── adversarial/
├── evals/
│   ├── questions.jsonl
│   ├── expected_evidence.jsonl
│   └── evaluate.py
├── docs/
│   ├── architecture.md
│   ├── privacy.md
│   ├── legal-source-policy.md
│   ├── evaluation.md
│   └── model-card.md
├── data/sample_contracts/
├── models/
├── LICENSE
└── requirements.txt
```

---

## Quickstart

### 1. Backend Setup
```bash
# Verify Python 3.11
python --version

# Install dependencies
pip install -r requirements.txt

# Run test suite
pytest
```

### 2. Run Offline Benchmark Harness
```bash
python evals/evaluate.py
```
