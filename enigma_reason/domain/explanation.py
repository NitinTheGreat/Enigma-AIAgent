"""Explanation domain models — structured, auditable reasoning projections.

Phase 6 introduces read-only explanation snapshots: immutable projections
of completed reasoning state that describe what the system believes, why,
why alternatives were rejected, and what would change its mind.

Phase 6.1 adds:
- ContributionDirection: quantifies factor influence direction
- Counterfactual: structured what-if projections from existing thresholds
- TemporalEvolution: summarises belief change dynamics
- contribution_score / contribution_direction on ExplanationSection

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
    - SUMMARY: What the system currently believes
    - SUPPORTING_EVIDENCE: Factors that reinforce the dominant hypothesis
    - CONTRADICTING_EVIDENCE: Factors that weaken or oppose it
    - WHY_UNKNOWN: Why the system remains undecided
    - CONFIDENCE_RATIONALE: How confidence was computed deterministically
    - WHAT_WOULD_CHANGE_MY_MIND: Missing evidence or thresholds needed
    - COUNTERFACTUALS: Structured what-if projections (Phase 6.1)
    - TEMPORAL_EVOLUTION: Belief change dynamics summary (Phase 6.1)
    """

    SUMMARY = "SUMMARY"
    SUPPORTING_EVIDENCE = "SUPPORTING_EVIDENCE"
    CONTRADICTING_EVIDENCE = "CONTRADICTING_EVIDENCE"
    WHY_UNKNOWN = "WHY_UNKNOWN"
    CONFIDENCE_RATIONALE = "CONFIDENCE_RATIONALE"
    WHAT_WOULD_CHANGE_MY_MIND = "WHAT_WOULD_CHANGE_MY_MIND"
    COUNTERFACTUALS = "COUNTERFACTUALS"
    TEMPORAL_EVOLUTION = "TEMPORAL_EVOLUTION"


class ContributionDirection(str, Enum):
    """Direction of an evidence factor's contribution to confidence."""

    SUPPORTING = "SUPPORTING"
    OPPOSING = "OPPOSING"
    NEUTRAL = "NEUTRAL"


class Counterfactual(BaseModel):
    """A structured what-if projection from existing thresholds.

    Each counterfactual describes a missing condition, its expected
    effect, and an estimated confidence delta — all derived from
    existing deterministic rules, never speculated.
    """

    missing_condition: str = Field(
        ..., min_length=5, max_length=200,
        description="The condition that is not currently met",
    )
    expected_effect: str = Field(
        ..., min_length=5, max_length=200,
        description="What would happen if the condition were met",
    )
    confidence_delta_estimate: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Estimated confidence change from existing thresholds",
    )


class TemporalEvolution(BaseModel):
    """Summary of how beliefs changed over reasoning iterations.

    Derived entirely from belief_velocity, belief_stability_score,
    and undecided_iterations — no new computation.
    """

    confidence_trend: str = Field(
        ..., description="rising | falling | flat",
    )
    belief_velocity_summary: str = Field(
        ..., description="slow | moderate | fast",
    )
    undecided_duration: int = Field(
        default=0, ge=0,
        description="Number of iterations system was undecided",
    )
    stability_label: str = Field(
        ..., description="stable | volatile | transitioning",
    )


class ExplanationSection(BaseModel):
    """A single typed section within an ExplanationSnapshot.

    Contains structured bullet points and references to the reasoning
    fields that informed this section.  No adjectives.  No speculation.

    Phase 6.1 adds:
        contribution_score: How much this factor contributed (0-1)
        contribution_direction: SUPPORTING | OPPOSING | NEUTRAL
        counterfactuals: Structured what-if projections (for COUNTERFACTUALS type)
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

    # Phase 6.1: Contribution scoring
    contribution_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Relative contribution strength of this factor (0-1)",
    )
    contribution_direction: ContributionDirection | None = Field(
        default=None,
        description="Direction of this factor's influence",
    )

    # Phase 6.1: Counterfactuals (only for COUNTERFACTUALS sections)
    counterfactuals: list[Counterfactual] = Field(
        default_factory=list,
        description="Structured what-if projections",
    )


class ExplanationSnapshot(BaseModel):
    """Immutable, structured explanation of a completed reasoning pass.

    This is a READ-ONLY projection — it never modifies belief state.
    It describes what the system believes, why, and what would change
    its mind, using only data already present in the reasoning state.

    Phase 6.1 adds temporal_evolution for belief dynamics summary.
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

    # Phase 6.1: Temporal belief evolution
    temporal_evolution: TemporalEvolution | None = Field(
        default=None,
        description="Summary of how beliefs evolved over iterations",
    )

    model_config = {"frozen": True}


# ── Role-Based View Filtering ───────────────────────────────────────────────


class ExplanationRole(str, Enum):
    """Consumer roles for filtered explanation views."""

    ANALYST = "ANALYST"      # Full detail
    MANAGER = "MANAGER"      # Summary + confidence + counterfactuals
    AUDITOR = "AUDITOR"      # Deterministic fields + thresholds only


# Section types visible per role
_ROLE_SECTIONS: dict[ExplanationRole, set[SectionType]] = {
    ExplanationRole.ANALYST: set(SectionType),  # all sections
    ExplanationRole.MANAGER: {
        SectionType.SUMMARY,
        SectionType.CONFIDENCE_RATIONALE,
        SectionType.COUNTERFACTUALS,
        SectionType.TEMPORAL_EVOLUTION,
    },
    ExplanationRole.AUDITOR: {
        SectionType.SUMMARY,
        SectionType.CONFIDENCE_RATIONALE,
        SectionType.WHAT_WOULD_CHANGE_MY_MIND,
    },
}


def filter_explanation_for_role(
    snapshot: ExplanationSnapshot,
    role: ExplanationRole,
) -> ExplanationSnapshot:
    """Produce a filtered view of an ExplanationSnapshot for a given role.

    This is a pure projection — no recomputation, no new data.
    The same snapshot is filtered to show only sections appropriate
    for the consumer's role.
    """
    allowed = _ROLE_SECTIONS.get(role, set(SectionType))
    filtered_sections = [
        s for s in snapshot.explanation_sections
        if s.section_type in allowed
    ]

    return ExplanationSnapshot(
        situation_id=snapshot.situation_id,
        dominant_hypothesis_id=snapshot.dominant_hypothesis_id,
        dominant_hypothesis_description=snapshot.dominant_hypothesis_description,
        dominant_confidence=snapshot.dominant_confidence,
        convergence_score=snapshot.convergence_score,
        belief_stability_score=snapshot.belief_stability_score,
        undecided=snapshot.undecided,
        iteration_count=snapshot.iteration_count,
        explanation_sections=filtered_sections,
        temporal_evolution=snapshot.temporal_evolution if role != ExplanationRole.AUDITOR else None,
    )
