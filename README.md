# Private Legal Triage System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](.python-version)
[![Node: 24](https://img.shields.io/badge/Node-24-green.svg)](.nvmrc)
[![Local-Only: Zero Egress](https://img.shields.io/badge/Privacy-100%25%20Local-success.svg)](docs/privacy.md)

A local-first, evidence-grounded legal document triage system for rental agreements (leases). It is **not** a general legal chatbot. It extracts contract contents locally, identifies key clauses, answers questions using verifiable verbatim evidence from the document, flags ambiguities and potential legal concerns, and explicitly abstains when the contract does not contain enough information.

**Every byte of data stays on your local device.** Zero cloud LLM calls, hosted embeddings, telemetry, or remote storage.

---

## Key Guarantees
- **Strictly Local**: Bound strictly to `127.0.0.1`. No external APIs or telemetry.
- **Verifiable Evidence**: Every factual claim requires a verbatim quote and exact character span from the original document.
- **Honest Abstention**: When the contract is silent on a topic, the system states that it cannot establish the answer rather than fabricating one from general model knowledge.
- **No Definitive Legal Rulings**: Never delivers definitive judgments of illegality. Questions regarding legal enforceability trigger an explicit limitation and escalation to a legal professional.
- **Prompt Injection Defense**: Ingested document text is isolated inside `<UNTRUSTED_DOCUMENT_EVIDENCE>` blocks and treated as passive evidence, never as instructions.

---

## Tech Stack
- **Backend**: Python 3.11, FastAPI (`127.0.0.1` only), Pydantic v2 schemas
- **PDF Extraction**: PyMuPDF (`fitz`) with page tracking and exact character offsets
- **Retrieval**: BM25 sparse ranking (strictly local; no external vector database)
- **Local Model**: Google Gemma edge model (Gemma 3n E2B / E4B or quantized GGUF)
- **Frontend**: React + TypeScript + Vite, local assets only, restrictive Content Security Policy
- **Testing**: pytest (unit, integration, safety, privacy, adversarial)

---

## Repository Structure

```
├── frontend/                     # React + TypeScript + Vite UI
│   ├── src/
│   │   ├── components/           # Document viewer, clause explorer, risk badges
│   │   ├── pages/                # Workspace, Overview, Privacy Controls
│   │   ├── api/                  # Local-only loopback API client
│   │   └── types/                # TypeScript schemas mirroring backend
│   └── package.json
├── backend/
│   └── app/
│       ├── main.py               # FastAPI entry point (127.0.0.1)
│       ├── config.py             # Local paths, loopback bindings, privacy flags
│       ├── api/                  # REST endpoints (upload, overview, clauses, ask)
│       ├── ingestion/            # PDF/TXT parsers, validator, normalizer
│       ├── document/             # Section detector, exact-span chunker, schemas
│       ├── retrieval/            # BM25 indexing, concept expansion, ranker
│       ├── agent/                # Local Gemma runtime, hardened prompts, analyzer
│       ├── legal_sources/        # Provenance boundary (contract vs statute)
│       ├── safety/               # Verbatim quote & claim validators, response guard
│       └── storage/              # SQLite session store & zero-trace cleanup
├── tests/
│   ├── unit/                     # Schema, parsing, and chunking tests
│   ├── integration/              # Pipeline ingestion & retrieval tests
│   ├── safety/                   # Hallucination & quote verification tests
│   ├── privacy/                  # Zero-egress network isolation tests
│   └── adversarial/              # Document prompt-injection & malformed PDF tests
├── evals/                        # Evaluation questions & benchmark runner
├── docs/                         # Architecture, privacy threat model, policies
├── data/sample_contracts/        # Synthetic rental agreements for testing
└── models/                       # Local model weights directory (gitignored)
```

---

## Getting Started

### 1. Prerequisites
- Python 3.11 (`python --version`)
- Node.js 20+ (`node --version`)

### 2. Backend Setup
```bash
# Install backend dependencies
python -m pip install -r requirements.txt

# Run test suite
pytest
```

### 3. Running Backend Locally
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

---

## Documentation
- [Architecture Overview](docs/architecture.md)
- [Privacy Policy & Threat Model](docs/privacy.md)
- [Legal Source Separation Policy](docs/legal-source-policy.md)
- [Evaluation Benchmark](docs/evaluation.md)
- [Model Card](docs/model-card.md)

---

## License
MIT License. See [LICENSE](LICENSE) for details.
