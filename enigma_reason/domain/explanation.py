"""Explanation domain models — structured, auditable reasoning projections.

Phase 6 introduces read-only explanation snapshots: immutable projections
of completed reasoning state that describe what the system believes, why,
why alternatives were rejected, and what would change its mind.

These models are NEVER used to modify reasoning state.  They are a
one-way projection.  No free text generation.  No speculation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from enigma_reason.foundation.identifiers import new_id


class SectionType(str, Enum):
    """Typed categories for explanation sections.

    Each section type has a strict semantic contract:
    - SUMMARY: What the system currently believes (1-2 sentences as bullets)
    - SUPPORTING_EVIDENCE: Factors that reinforce the dominant hypothesis
    - CONTRADICTING_EVIDENCE: Factors that weaken or oppose it
    - WHY_UNKNOWN: Why the system remains undecided (if applicable)
    - CONFIDENCE_RATIONALE: How confidence was computed deterministically
    - WHAT_WOULD_CHANGE_MY_MIND: Missing evidence or thresholds needed
    """

    SUMMARY = "SUMMARY"
    SUPPORTING_EVIDENCE = "SUPPORTING_EVIDENCE"
    CONTRADICTING_EVIDENCE = "CONTRADICTING_EVIDENCE"
    WHY_UNKNOWN = "WHY_UNKNOWN"
    CONFIDENCE_RATIONALE = "CONFIDENCE_RATIONALE"
    WHAT_WOULD_CHANGE_MY_MIND = "WHAT_WOULD_CHANGE_MY_MIND"


class ExplanationSection(BaseModel):
    """A single typed section within an ExplanationSnapshot.

    Contains structured bullet points and references to the reasoning
    fields that informed this section.  No adjectives.  No speculation.
    """

    section_type: SectionType
    title: str = Field(..., min_length=3, max_length=100)
    bullet_points: list[str] = Field(
        default_factory=list,
        description="Short, factual bullet points.  No adjectives.",
    )
    referenced_fields: list[str] = Field(
        default_factory=list,
        description="Names of reasoning fields that informed this section.",
    )


class ExplanationSnapshot(BaseModel):
    """Immutable, structured explanation of a completed reasoning pass.

    This is a READ-ONLY projection — it never modifies belief state.
    It describes what the system believes, why, and what would change
    its mind, using only data already present in the reasoning state.
    """

    explanation_id: str = Field(default_factory=lambda: str(new_id()))
    situation_id: str
    dominant_hypothesis_id: str | None = Field(
        default=None,
        description="ID of the leading hypothesis, or None if undecided",
    )
    dominant_hypothesis_description: str | None = Field(
        default=None,
        description="Description of the leading hypothesis",
    )
    dominant_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    convergence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    belief_stability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    undecided: bool = Field(
        default=True,
        description="True if UNKNOWN is dominant or no hypothesis has converged",
    )
    iteration_count: int = Field(default=0)
    explanation_sections: list[ExplanationSection] = Field(
        default_factory=list,
        description="Ordered list of typed explanation sections",
    )

    model_config = {"frozen": True}
