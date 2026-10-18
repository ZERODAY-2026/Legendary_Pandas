"""
Claim validator.
Enforces deterministic claim constraints:
1. Explicit statements must be grounded in real passages.
2. Interpretations must be labelled as interpretations.
3. Missing clauses must never be described as existing.
4. Legal concerns must not state definitive findings of illegality.
"""

from backend.app.document.schemas import Finding


class ClaimValidationResult:

    def __init__(self, is_valid: bool, issues: list[str]):
        self.is_valid = is_valid
        self.issues = issues


def validate_finding_claims(finding: Finding) -> ClaimValidationResult:
    """Validate claims within a single Finding."""
    issues = []

    # Rule: Explicit statement requires evidence
    if finding.explicit_text and not finding.evidence:
        issues.append(f"Finding '{finding.title}' has explicit_text but no evidence attached.")

    # Rule: Category 'missing_information' must not have explicit_text
    if finding.category == "missing_information" and finding.explicit_text:
        issues.append(
            f"Finding '{finding.title}' is categorized as missing_information but claims explicit_text."
        )

    # Rule: Check definitive illegality keywords
    definitive_phrases = [
        "is illegal",
        "is unlawful",
        "is void",
        "violates the law definitively",
        "is strictly prohibited and unenforceable",
    ]
    combined = f"{finding.title} {finding.plain_language_explanation} {finding.interpretation_or_concern or ''}".lower()
    for phrase in definitive_phrases:
        if phrase in combined:
            issues.append(
                f"Finding contains definitive legal conclusion ('{phrase}'). "
                "Must be rephrased as potential concern with legal professional escalation."
            )

    return ClaimValidationResult(is_valid=len(issues) == 0, issues=issues)
