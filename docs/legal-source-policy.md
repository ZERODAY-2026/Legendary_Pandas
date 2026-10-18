# Legal Source Policy & Provenance Boundary

## 1. Principle of Strict Source Separation
In legal document triage, conflating the user's uploaded contract with external legislation or judicial precedent leads to disastrous errors (e.g., treating a statutory tenant protection right as if it were an explicit clause granted by the landlord, or vice versa).

The system enforces three distinct source types:
1. **`contract`**: The uploaded rental agreement (active in MVP).
2. **`legislation`**: Statutory tenancy laws (out of scope for MVP; reserved for future retrieval).
3. **`court_judgment`**: Case law and judicial precedents (out of scope for MVP; reserved for future retrieval).

---

## 2. Immutable Ingestion-Time Assignment
- **Rule 1**: The source type is assigned strictly at the point of ingestion by the deterministic system.
- **Rule 2**: The language model is **never** permitted to infer, predict, change, or assign a source type.
- **Rule 3**: Any chunk or evidence item without a validated `SourceType` is rejected by the schema validator.

---

## 3. Non-Conflation Guarantees
- An explicit statement in a `Finding` must point directly to a `contract` evidence item containing a verbatim quote.
- External legislation and judgments must never be used as if they were part of the uploaded contract.
- If a contract is silent on a matter (e.g., subletting rules or pet policies), the system must explicitly state:
  > *"The contract does not provide enough information on this topic."*
  It must **never** fill silence using general LLM pre-training knowledge.

---

## 4. Legal Uncertainty & Mandatory Escalation Policy
- **No Definitive Rulings**: The system must never assert that a clause is definitively legal, illegal, valid, or void. Enforceability is a jurisdiction-dependent judicial determination.
- **Potential Concerns**: Clauses that appear punitive (e.g., excessive late fees, complete waiver of landlord habitability obligations) must be categorized strictly as `"potential_legal_concern"`.
- **Mandatory Escalation**: Whenever a question asks for a legal opinion (e.g., *"Is this enforceable?"*, *"Can I sue?"*), the system must set status to `"escalation_required"` and direct the user to a licensed attorney or legal aid clinic.
