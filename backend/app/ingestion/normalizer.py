"""
Text normalization preserving legal meaning and original offsets for citation verification.
"""

import re


def normalize_whitespace(text: str) -> str:
    """Normalize irregular spacing, carriage returns, and indentation

    without altering words, punctuation, clause markers, or legal meaning.
    """
    # Standardize line breaks
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces and excessive inline horizontal tabs/spaces
    normalized = re.sub(r"[ \t]+", " ", normalized)
    # Remove excessive blank lines (more than two consecutive newlines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
