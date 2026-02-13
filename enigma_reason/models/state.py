"""Pydantic model for the evolving situation state maintained by the reasoning layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from enigma_reason.models.signal import IncomingSignal


class ThreatLevel(str, Enum):
    NONE = "none"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class SituationState(BaseModel):
    """Accumulating state that the reasoning graph evolves over time."""

    state_id: UUID = Field(default_factory=uuid4)
    threat_level: ThreatLevel = Field(default=ThreatLevel.NONE)
    active_signals: list[IncomingSignal] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict, description="Derived situational context")
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    reasoning_trace: list[str] = Field(default_factory=list, description="Step-by-step reasoning log")
