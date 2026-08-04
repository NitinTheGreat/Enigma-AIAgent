"""Reasoning domain models — deterministic, explainable observations.

Phase 4 introduces numerical reasoning without LLMs: confidence levels,
trend detection, and reasoning snapshots.  These are pure observations
derived from evidence and temporal metrics.  No opinions, no risk labels.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Trend(str, Enum):
    """Trajectory of a situation's activity over time."""

    ESCALATING = "escalating"
    STABLE = "stable"
    DEESCALATING = "deescalating"


class SituationReasoningSnapshot(BaseModel):
    """Immutable reasoning observation of a situation at a point in time.

    All fields are deterministically derived from evidence, temporal metrics,
    and source diversity.  Nothing here is subjective.
    """

    situation_id: str
    evidence_count: int = Field(..., description="Total evidence items")
    event_rate: float = Field(..., description="Events per minute")
    burst_detected: bool = Field(..., description="Whether burst is currently active")
    quiet_detected: bool = Field(..., description="Whether situation is quiet")
    confidence_level: float = Field(
        ..., ge=0.0, le=1.0,
        description="Deterministic confidence in the situation's significance (0–1)",
    )
    trend: Trend = Field(..., description="Current trajectory")
    source_diversity: int = Field(..., description="Count of distinct signal sources")
    mean_anomaly_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Average anomaly score across all evidence",
    )
    abstained_evidence_count: int = Field(
        default=0,
        description="Evidence items where the sensor declined to assign a class",
    )
    abstained_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Share of evidence the sensor abstained on. High values mean the "
            "situation rests on detections the sensor itself would not stand "
            "behind, which is a reason for the reasoner to remain undecided."
        ),
    )

    model_config = {"frozen": True}
