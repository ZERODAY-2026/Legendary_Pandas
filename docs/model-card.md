# Model Card: Local Gemma Legal Triage Engine

## 1. Model Details
- **Architecture**: Google Gemma 3n / Gemma 2B Edge Variant (`gemma-3-e2b-it` or `gemma-2-2b-it`)
- **Developer**: Google DeepMind (weights adapted for local offline inference)
- **Quantization**: 4-bit quantized (Q4_K_M GGUF) or 8-bit quantized (Q8_0)
- **Local Inference Runtime**: Local llama-cpp / PyTorch edge inference engine
- **Context Length**: 4,096 tokens
- **Hardware Footprint**:
  - Model weights on disk: ~1.5 GB - 2.8 GB
  - Active RAM during generation: ~3.5 GB - 5.5 GB
  - Acceleration: Optimized for CPU (AVX2), Apple Silicon (Metal), or NVIDIA CUDA

---

## 2. Intended Use
- **Primary Domain**: Residential and commercial rental agreements (leases) uploaded by tenants, students, or landlords.
- **Tasks Supported**:
  - Plain-language explanation of complex legal clauses.
  - Identification of key obligations, rights, deposit rules, notice periods, and maintenance terms.
  - Highlighting ambiguities, unusual indemnity clauses, and excessive penalty fees.
  - Explicit abstention when terms are omitted or silent.

---

## 3. Out-of-Scope & Prohibited Uses
- **NOT Legal Advice**: This system does not establish an attorney-client relationship.
- **NO Definitive Enforceability Judgments**: Cannot declare whether a clause is legally enforceable in a specific jurisdiction.
- **NO Autonomous Litigation Filings**: Must not be used to draft legal pleadings without human attorney supervision.

---

## 4. Grounding & Hallucination Mitigations
- Ingested contract text is placed inside `<UNTRUSTED_DOCUMENT_EVIDENCE>` tags.
- Generation temperature is pinned to `0.0` (deterministic greedy decoding).
- Every finding is post-validated deterministically: quotes must be exact substrings of raw text, or the finding is flagged and rejected.
