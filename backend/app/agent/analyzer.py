"""
Clause and contract analyzer.
Analyzes retrieved evidence, structures findings, and ensures compliance with schema constraints.
"""

from backend.app.agent.gemma_runtime import GemmaRuntime
from backend.app.agent.prompts import SYSTEM_INSTRUCTION, PROMPT_TEMPLATE, build_evidence_block
from backend.app.document.schemas import Chunk, Finding, QueryResponse


class ContractAnalyzer:
    """Orchestrates generation of findings from retrieved chunks."""

    def __init__(self, runtime: GemmaRuntime):
        self.runtime = runtime

    def analyze_question(
        self,
        question: str,
        retrieved_chunks: list[Chunk],
    ) -> QueryResponse:
        """Run analysis through local Gemma or fallback to insufficient evidence."""
        if not retrieved_chunks:
            return QueryResponse(
                answer="The provided rental agreement does not contain sufficient information to answer this question.",
                status="insufficient_evidence",
                findings=[
                    Finding(
                        title="Insufficient Evidence in Agreement",
                        category="missing_information",
                        plain_language_explanation=(
                            "The document was analyzed, but no relevant clauses or provisions "
                            f"were found regarding: '{question}'."
                        ),
                    )
                ],
                evidence_used=[],
            )

        # Full prompt assembly and local generation in Phase 5
        return QueryResponse(
            answer="Analysis pending model execution.",
            status="insufficient_evidence",
            findings=[],
            evidence_used=[],
        )
