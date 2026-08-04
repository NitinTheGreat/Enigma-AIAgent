"""Canonical Signal model — the contract between ML service and reasoning layer.

A Signal is a *claim with uncertainty*, not a fact.  It represents what an
upstream detector believes it observed, together with a confidence score.
This model is deliberately strict: no Dict[str, Any] blobs, no optional
free-text overrides.  Every field is explicit and validated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from enigma_reason.domain.enums import EntityKind, SignalType


# ── Entity Reference ─────────────────────────────────────────────────────────

class EntityRef(BaseModel):
    """Structured reference to the entity this signal is about."""

    kind: EntityKind
    identifier: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique identifier within its kind (username, device-id, etc.)",
    )

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.identifier}"


# ── Signal ───────────────────────────────────────────────────────────────────

class Signal(BaseModel):
    """An incoming anomaly signal from the external ML detection service.

    Immutable after creation.  Validated at the boundary so downstream
    code never has to re-check field constraints.
    """

    signal_id: UUID = Field(..., description="Unique signal identifier assigned by the ML service")
    timestamp: datetime = Field(..., description="When the anomaly was observed (must be UTC-aware)")
    signal_type: SignalType = Field(..., description="Controlled classification label")
    entity: Optional[EntityRef] = Field(
        default=None,
        description="The entity this signal pertains to, if identifiable",
    )
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How anomalous the detector considers this event, defined as "
            "1 - P(normal). 0 means the detector is sure the flow is benign, "
            "1 means it is sure the flow is not. This is NOT the classifier's "
            "confidence in its chosen label; see predicted_class_confidence."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detector's confidence in its own classification (0 = guessing, 1 = certain)",
    )
    predicted_class_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum class probability, meaning how sure the detector is about "
            "the label it chose. Carries no information about whether the flow "
            "is benign, because a confidently benign flow scores high here too."
        ),
    )
    predictive_entropy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Shannon entropy of the class distribution normalised by log of the "
            "class count, so 0 is a one-hot prediction and 1 is uniform. A proxy "
            "for detector uncertainty, reserved for Level 4 selective prediction."
        ),
    )
    features: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Short descriptive feature tags that contributed to detection",
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Identifier of the detection pipeline or sensor",
    )
    abstained: bool = Field(
        default=False,
        description=(
            "True when the sensor declined to assign a class because its "
            "calibrated confidence fell below the validation-fitted threshold. "
            "An abstained signal is still evidence that something occurred; it "
            "is not a claim about what."
        ),
    )
    calibrated_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Temperature-scaled confidence that drove the abstention decision",
    )

    model_config = {"frozen": True}

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, v: datetime) -> datetime:
        # Auto-attach UTC if the timestamp is naive (common in ML pipelines)
        if v.tzinfo is None:
            v = v.replace(tzinfo=__import__("datetime").timezone.utc)
        return v

    @field_validator("features")
    @classmethod
    def features_must_be_short(cls, v: list[str]) -> list[str]:
        for tag in v:
            if len(tag) > 64:
                raise ValueError(f"feature tag too long ({len(tag)} chars, max 64): {tag[:40]}…")
        return v
