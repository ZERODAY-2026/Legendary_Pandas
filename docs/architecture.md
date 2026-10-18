# Architecture Overview: Private Legal Triage System

## 1. System Vision & Local-First Philosophy
The Private Legal Triage System is an offline, edge-native legal triage tool engineered specifically for analyzing rental agreements (leases). It is **not** a general legal chatbot and **never** dispenses definitive legal judgments or legal advice. Its core purpose is to extract factual clauses, summarize tenant obligations and rights in plain language, flag ambiguities and potential legal concerns, and strictly abstain when information is not present in the document.

All processing—text extraction, clause chunking, sparse indexing, prompting, local Gemma inference, and temporary file management—runs exclusively on the user's local machine (bound to `127.0.0.1`).

```
+-----------------------------------------------------------------------------------+
|                               LOCAL DEVICE BOUNDARY                               |
|                                                                                   |
|  +---------------------+        HTTP / Loopback (127.0.0.1)      +-------------+  |
|  | React + TS Frontend | <=====================================> | FastAPI App |  |
|  | (Vite, Local Assets)|                                         +------+------+  |
|  +---------------------+                                                |         |
|                                                                         |         |
|         +---------------------------------------------------------------+         |
|         |                                                                         |
|         v                                                                         |
|  +--------------+    +----------------+    +----------------+    +-------------+  |
|  |  Ingestion   | -> |  Segmentation  | -> |  BM25 Ranking  | -> | Local Gemma |  |
|  | (PDF/TXT)    |    |  & Exact Span  |    |  & Search      |    | (Gemma 3n)  |  |
|  +--------------+    +----------------+    +----------------+    +------+------+  |
|                                                                         |         |
|                                                                         v         |
|                                                                  +-------------+  |
|                                                                  | Safety &    |  |
|                                                                  | Citation    |  |
|                                                                  | Guardrails  |  |
|                                                                  +-------------+  |
+-----------------------------------------------------------------------------------+
                                 ||
                        [NO NETWORK EGRESS]
```

---

## 2. Component Pipeline

### A. Document Ingestion (`backend/app/ingestion/`)
- **PDF Parser (`pdf_parser.py`)**: Uses local PyMuPDF (`fitz`) to extract page-by-page text while recording exact character start and end offsets (`span_start`, `span_end`). Detects empty, encrypted, corrupted, and image-only PDFs, throwing deterministic errors.
- **TXT Parser (`txt_parser.py`)**: Reads UTF-8/Latin-1 text files, mapping contents to logical page spans.
- **Validator (`validator.py`)**: Validates MIME type, magic bytes (`%PDF-`), file size limits (<= 25 MB), and extension checks.
- **Normalizer (`normalizer.py`)**: Standardizes linebreaks and excessive whitespace without changing legal wording, punctuation, or clause markers. Original text is preserved unmodified for citation verification.

### B. Segmentation & Chunking (`backend/app/document/`)
- **Section Detector (`section_detector.py`)**: Detects headings, numbered clauses (e.g., "1. TERM", "Clause 4.2"), and structural divisions.
- **Chunker (`chunker.py`)**: Segments documents by legal boundaries first. Bounded sub-chunking is applied only to exceptionally long sections.
- **Span Invariant**: Every `Chunk` records `document_id`, `source_type`, `page_start`, `page_end`, `clause_id`, `span_start`, and `span_end`. Crucially:
  $$\text{raw\_text}[\text{span\_start} : \text{span\_end}] == \text{chunk.text}$$

### C. Sparse Retrieval (`backend/app/retrieval/`)
- **Index (`index.py`)**: Implements an in-memory BM25Okapi sparse index over the active document's chunks. Zero external embedding APIs or neural vector stores.
- **Search & Query Expansion (`search.py`)**: Tokenizes user queries and expands legal concepts (e.g., *termination* $\to$ *notice, breach, expiry*). Computes BM25 relevance scores.
- **Ranker (`ranker.py`)**: Applies minimum score thresholds, deduplicates overlapping passages, and fits relevant passages into a strict token budget ($\le 2048$ tokens).
- **Silence & Abstention**: If no chunk exceeds the minimum relevance threshold, retrieval immediately halts and outputs an explicit "insufficient evidence" state.

### D. Local Gemma Inference (`backend/app/agent/`)
- **Runtime (`gemma_runtime.py`)**: Local-only execution of Google Gemma models (specifically Gemma 3n E2B/E4B or quantized edge GGUF). Memory allocation, context bounds, and greedy generation ($\text{temperature} = 0.0$) are encapsulated here.
- **Prompt Injection Hardening (`prompts.py`)**: Document passages are passed strictly inside `<UNTRUSTED_DOCUMENT_EVIDENCE>` tags. System instructions explicitly treat document content as untrusted evidence rather than operational commands.
- **Analyzer (`analyzer.py`)**: Prompts Gemma to generate structured JSON containing:
  - Plain language summary
  - Extraction of rights, obligations, deadlines, and penalties
  - Explicit quotes vs. interpretive notes
  - Abstention on missing information

### E. Safety & Deterministic Verification (`backend/app/safety/`)
- **Citation Validator (`citation_validator.py`)**: Verifies that every cited quote is an exact verbatim substring of the original extracted document text. Rejects hallucinations or misattributed page/section citations.
- **Claim Validator (`claim_validator.py`)**: Enforces deterministic schema rules on every `Finding`:
  1. An explicit statement must be backed by a verbatim quote.
  2. Missing information cannot cite non-existent clauses.
  3. No finding may state a definitive verdict of illegality or unenforceability.
- **Response Guard (`response_guard.py`)**: Detects questions seeking definitive legal advice (e.g., "is this enforceable?"), forces the status to `escalation_required`, and appends an attorney consultation mandate.

### F. Legal Source Separation (`backend/app/legal_sources/`)
- Explicitly isolates three types: `contract`, `legislation`, and `court_judgment`.
- Assigned deterministically at ingestion. The model is **never** permitted to infer or reassign the source type.
- External legislation and judgments are out of scope for the MVP but architectural boundaries prevent future cross-contamination.

### G. Local Storage & Privacy (`backend/app/storage/`)
- Local SQLite database and ephemeral filesystem storage only.
- Complete, user-driven session purging (`cleanup.py`).
- Strict audit: logging configuration strips document text, queries, names, and generated answers.
