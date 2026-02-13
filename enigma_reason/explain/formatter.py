"""ExplanationFormatter — optional LLM-based phrasing of explanations.

This formatter takes a structured ExplanationSnapshot and produces
human-readable text.  The LLM is used ONLY to rephrase existing
structured content — it cannot invent new claims, add speculation,
or reference data not present in the snapshot.

Usage:
    formatter = ExplanationFormatter(llm_factory)
    text = formatter.format(explanation_snapshot)

This component is FULLY OPTIONAL and replaceable.  The system works
without it — ExplanationSnapshot is already structured and auditable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from enigma_reason.domain.explanation import ExplanationSnapshot, SectionType

logger = logging.getLogger(__name__)

LLMFactory = Callable[[], Any]

_FORMAT_PROMPT = """You are a technical report formatter.  Your task is to rephrase
the following structured explanation into clear, professional prose.

STRICT RULES:
- Use ONLY the information provided below
- Do NOT add facts, speculation, or adjectives not present in the input
- Do NOT invent new claims or evidence
- Keep tone neutral and factual
- Use short paragraphs

Situation ID: {situation_id}
Undecided: {undecided}
Dominant Hypothesis: {dominant_desc}
Dominant Confidence: {dominant_conf:.2f}
Convergence Score: {convergence:.2f}
Iterations: {iterations}

Sections:
{sections_text}

Produce a brief, professional explanation (3-8 paragraphs).  Start with the
assessment status.  End with what would change the assessment."""


class ExplanationFormatter:
    """Optional LLM-powered formatter for ExplanationSnapshot.

    The LLM receives ONLY structured data already present in the snapshot.
    It cannot invent new claims.  If the LLM fails, a deterministic
    fallback is used.
    """

    def __init__(self, llm_factory: LLMFactory) -> None:
        self._llm_factory = llm_factory

    def format(self, snapshot: ExplanationSnapshot) -> str:
        """Format an ExplanationSnapshot into human-readable text.

        Falls back to deterministic plain-text formatting on LLM failure.
        """
        try:
            return self._format_with_llm(snapshot)
        except Exception as exc:
            logger.warning("LLM formatting failed: %s — using fallback", exc)
            return self.format_plain(snapshot)

    def _format_with_llm(self, snapshot: ExplanationSnapshot) -> str:
        """Use LLM to produce polished prose from structured data."""
        sections_text = self._sections_to_text(snapshot)

        prompt = _FORMAT_PROMPT.format(
            situation_id=snapshot.situation_id,
            undecided=snapshot.undecided,
            dominant_desc=snapshot.dominant_hypothesis_description or "None",
            dominant_conf=snapshot.dominant_confidence,
            convergence=snapshot.convergence_score,
            iterations=snapshot.iteration_count,
            sections_text=sections_text,
        )

        llm = self._llm_factory()
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()

    @staticmethod
    def _sections_to_text(snapshot: ExplanationSnapshot) -> str:
        """Convert sections to plain text for LLM input."""
        parts = []
        for section in snapshot.explanation_sections:
            parts.append(f"### {section.title} [{section.section_type.value}]")
            for bp in section.bullet_points:
                parts.append(f"  - {bp}")
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def format_plain(snapshot: ExplanationSnapshot) -> str:
        """Deterministic plain-text formatting without LLM.

        This is always available as a fallback and produces consistent,
        structured output suitable for logs, APIs, or debugging.
        """
        lines = [f"Explanation for situation {snapshot.situation_id}"]
        lines.append("=" * 50)

        if snapshot.undecided:
            lines.append("STATUS: UNDECIDED")
        else:
            lines.append("STATUS: CONVERGED")

        lines.append(f"Dominant: {snapshot.dominant_hypothesis_description or 'None'}")
        lines.append(f"Confidence: {snapshot.dominant_confidence:.2f}")
        lines.append(f"Convergence: {snapshot.convergence_score:.2f}")
        lines.append(f"Stability: {snapshot.belief_stability_score:.2f}")
        lines.append(f"Iterations: {snapshot.iteration_count}")
        lines.append("")

        for section in snapshot.explanation_sections:
            lines.append(f"--- {section.title} [{section.section_type.value}] ---")
            for bp in section.bullet_points:
                lines.append(f"  • {bp}")
            lines.append("")

        return "\n".join(lines)
