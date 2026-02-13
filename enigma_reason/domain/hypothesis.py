"""Hypothesis — a structured, neutral conjecture about a situation.

A hypothesis is NOT an explanation or a decision.  It is a short, factual
statement about what might be happening, paired with a confidence score
that is adjusted deterministically by the reasoning graph.

Phase 5 introduces hypotheses as the unit of reasoning inside the
LangGraph orchestration loop.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from enigma_reason.foundation.identifiers import new_id


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

    model_config = {"frozen": False}
