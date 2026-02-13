"""Hypothesis — a structured, neutral conjecture about a situation.

A hypothesis is NOT an explanation or a decision.  It is a short, factual
statement about what might be happening, paired with a confidence score
that is adjusted deterministically by the reasoning graph.

Phase 5 introduces hypotheses as the unit of reasoning inside the
LangGraph orchestration loop.

Phase 5.1 adds belief inertia fields (velocity, acceleration) to resist
premature convergence and enforce epistemic discipline.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from enigma_reason.foundation.identifiers import new_id


# ── Constants ────────────────────────────────────────────────────────────────

UNKNOWN_HYPOTHESIS_ID = "UNKNOWN"
UNKNOWN_DESCRIPTION = "Insufficient information to determine cause"


class HypothesisStatus(str, Enum):
    """Lifecycle of a hypothesis within a reasoning loop."""

    ACTIVE = "active"
    PRUNED = "pruned"
    CONVERGED = "converged"


class Hypothesis(BaseModel):
    """A single conjecture about a situation.

    Created by the LLM, evaluated deterministically, pruned or converged
    by the reasoning graph.  Descriptions must be short and factual —
    no natural-language explanations.

    Phase 5.1 fields:
        belief_velocity: Rate of confidence change per iteration.
        belief_acceleration: Rate of velocity change (dampening factor).
        dominant_iterations: How many consecutive iterations this hypothesis
                            has been the highest-confidence active hypothesis.
    """

    hypothesis_id: str = Field(default_factory=lambda: str(new_id()))
    description: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Short, neutral factual statement",
    )
    confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Current confidence in this hypothesis (adjusted by evaluation)",
    )
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Signal IDs that support this hypothesis",
    )
    contradicting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Signal IDs that contradict this hypothesis",
    )
    status: HypothesisStatus = Field(
        default=HypothesisStatus.ACTIVE,
        description="Current lifecycle state",
    )

    # ── Phase 5.1: Belief inertia ────────────────────────────────────────
    belief_velocity: float = Field(
        default=0.0,
        description="Rate of confidence change per iteration (dampened by inertia)",
    )
    belief_acceleration: float = Field(
        default=0.0,
        description="Rate of velocity change — positive = accelerating belief",
    )
    dominant_iterations: int = Field(
        default=0,
        description="Consecutive iterations as highest-confidence active hypothesis",
    )

    model_config = {"frozen": False}


def make_unknown_hypothesis(confidence: float = 0.4) -> dict:
    """Create the permanent UNKNOWN hypothesis dict.

    UNKNOWN competes with all others and can never be pruned.
    It starts with moderate confidence to resist premature belief.
    """
    return {
        "hypothesis_id": UNKNOWN_HYPOTHESIS_ID,
        "description": UNKNOWN_DESCRIPTION,
        "confidence": confidence,
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "status": "active",
        "belief_velocity": 0.0,
        "belief_acceleration": 0.0,
        "dominant_iterations": 0,
    }
