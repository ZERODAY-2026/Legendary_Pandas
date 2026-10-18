"""
Response guard.
Detects questions requiring definitive legal judgment, triggers legal escalation,
and applies post-generation safety barriers.
"""

import re
from backend.app.document.schemas import QueryResponse

LEGAL_ADVICE_TRIGGERS = [
    r"\bis (?:this|the clause|it) (?:legally )?enforceable\b",
    r"\bis (?:this|it) legal\b",
    r"\bis (?:this|it) illegal\b",
    r"\bcan i sue\b",
    r"\bwill i win in court\b",
    r"\bdoes this violate the law\b",
    r"\bis this contract void\b",
]


def requires_legal_escalation(question: str) -> bool:
    """Check if the user's question demands a definitive legal judgment or advice."""
    for pattern in LEGAL_ADVICE_TRIGGERS:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False


def apply_response_guard(response: QueryResponse, question: str) -> QueryResponse:
    """Apply safety gates to QueryResponse before returning to client."""
    if requires_legal_escalation(question):
        response.status = "escalation_required"
        escalation_note = (
            "\n\n[MANDATORY NOTICE: Whether a clause is legally enforceable depends on applicable "
            "state/local tenancy statutes and judicial interpretation. This local triage tool provides "
            "factual analysis of the contract text and cannot render formal legal judgments. "
            "Please consult a licensed tenant rights attorney or legal aid organization.]"
        )
        if escalation_note not in response.answer:
            response.answer += escalation_note

    return response
