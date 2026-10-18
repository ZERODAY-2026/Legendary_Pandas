# Legal Source Policy & Provenance Boundary

## 1. Source Classification Hierarchy

The Private Legal Triage system enforces a strict categorical separation among legal materials. At ingestion, every ingested document is assigned exactly one immutable `SourceType`:

```
                    ┌────────────────────────────┐
                    │      Source Registry       │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│     contract     │    │   legislation    │    │  court_judgment  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
  (Active Upload)         (Statutory Rules)       (Case Law / Ratio)
```

1. **`contract`**:
   - The primary subject of analysis uploaded by the user (rental agreement, lease, tenancy addendum).
   - Contains party obligations, terms, rent, deposit rules, and notice periods.
   - **MVP Scope**: The MVP solely evaluates documents with `SourceType.CONTRACT`.
2. **`legislation`**:
   - Statutes, acts, and statutory instruments (e.g., Housing Act, Residential Tenancies Act, Landlord and Tenant Act).
   - Provides external regulatory bounds (e.g., statutory cap on security deposits, required notice periods).
   - *Architecture ready: Not ingested in MVP, but schema-separated from contracts.*
3. **`court_judgment`**:
   - Judicial rulings, precedent, and adjudications.
   - Interprets statutory provisions and contractual ambiguity.
   - *Architecture ready: Separated from both contracts and legislation.*

---

## 2. Inviolable Source Separation Rules

1. **Ingestion-Time Assignment Only**:
   - `SourceType` is assigned by ingestion logic or user registration.
   - The language model is **never** permitted to infer, reclassify, or alter a document's source type.
2. **No Source Conflation (Rule 5)**:
   - Legislation and court judgments must **never** be cited or referenced as if they were terms of the uploaded contract.
   - Example violation: Presenting a 30-day statutory notice rule as a clause in the tenant's lease when the lease itself says 14 days.
   - Any finding regarding the contract must only cite evidence with `source_type == SourceType.CONTRACT`.
3. **Distinction Between Contract Evidence and External Law**:
   - Findings on what the lease says must point exclusively to `contract` passages.
   - Potential conflicts with external law (when legislation is added) must be framed as comparative findings, never as internal lease clauses.

---

## 3. Grounding and Escalation Policy

1. **Grounding Priority**:
   - The system is grounded in verifiable text. If a fact cannot be established through a verbatim passage in the contract, the system must report that the contract does not provide sufficient information.
2. **Deterministic Legal Safety**:
   - The system must never emit a definitive legal conclusion of illegality (e.g., "Clause 4 is illegal" or "This waiver is unenforceable").
   - Instead, the system frames issues as:
     - *"Potential concern regarding deposit limit"*
     - *"The agreement does not explicitly state notice requirements"*
     - *"Consultation with a qualified tenancy lawyer is advised to confirm local statutory compliance"*
3. **Escalation Triggers**:
   - Questions requesting definitive legal judgments (enforceability, litigation outcomes, legal validity) trigger mandatory legal escalation notices.
