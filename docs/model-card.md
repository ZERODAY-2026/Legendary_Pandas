# Model Card: Local Gemma Runtime Specification

## 1. Model Overview

- **Model Family**: Google Gemma Edge Architecture (Gemma 3n / Gemma 2B Class)
- **Target Variants**:
  - `gemma-2-2b-it` (GGUF quantized Q4_K_M or Q4_0)
  - `gemma-3n` E2B / E4B edge variant
- **Execution Mode**: 100% Local, In-Process or Loopback Local Daemon (strictly no cloud fallback)
- **Primary Use Case**: Information extraction, clause summarization, right/obligation decomposition, and structured legal triage of rental agreements.

---

## 2. Hardware Resource Profile & Benchmarks

Benchmarked on host machine architecture:
- **Processor**: Multi-core x86_64 CPU
- **Accelerator**: CPU-only execution (No CUDA detected on host)
- **System Memory**: ~16 GB RAM (15.7 GB physical memory)

### Target Operational Budgets

| Parameter | 2B Class (Q4_K_M) | 4B Class (Q4_K_M) |
| :--- | :--- | :--- |
| **Model Disk Size** | ~1.6 GB | ~2.6 GB |
| **RAM Footprint (Weights)** | ~2.1 GB | ~3.4 GB |
| **Peak Runtime RAM** | ~3.0 GB | ~4.5 GB |
| **Context Window** | 4,096 tokens | 4,096 tokens |
| **Evidence Context Budget** | 4,000 characters | 4,000 characters |
| **Target Prompt Speed** | 25-45 tokens/sec | 15-25 tokens/sec |
| **Target Generation Speed**| 8-15 tokens/sec | 4-8 tokens/sec |

---

## 3. Input Envelope & Prompting Architecture

All document passages are quarantined in an untrusted evidence envelope:

```
[SYSTEM]
You are a local legal triage assistant. You extract verifiable facts and identify potential concerns.
RULES:
1. Base all statements exclusively on UNTRUSTED EVIDENCE PASSAGES below.
2. If evidence is insufficient, set status to "insufficient_evidence".
3. Never declare a clause "illegal" or "unenforceable". Frame legal issues as potential concerns for counsel.
4. Output valid JSON matching the triage schema.

[UNTRUSTED EVIDENCE PASSAGES]
{evidence_blocks}

[USER QUESTION]
{question}
```

---

## 4. Honest Limitations & Non-Goals

1. **Not Legal Advice**: Gemma is not an attorney and cannot provide legal representation, litigation predictions, or binding enforceability opinions.
2. **Local CPU Inference Latency**: On CPU-only systems, generation latency is bounded by single-core memory bandwidth. Context budgets must be kept tight (<= 5 retrieved chunks).
3. **Out-of-Scope Uses**: Do not use this model for automated contract signing decisions, drafting commercial merger agreements, or replacing licensed tenant advocacy.
