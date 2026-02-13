"""Pydantic model for decisions produced by the reasoning layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    ALERT = "alert"
    ESCALATE = "escalate"
    MITIGATE = "mitigate"
    MONITOR = "monitor"
    IGNORE = "ignore"


class DecisionOutput(BaseModel):
    """A decision produced by the agentic reasoning graph, streamed to the frontend."""

    decision_id: UUID = Field(default_factory=uuid4)
    action: ActionType = Field(..., description="Recommended action")
    summary: str = Field(..., description="Human-readable decision summary")
    explanation: str = Field(default="", description="Step-by-step reasoning explanation")
    confidence: float = Field(..., ge=0.0, le=1.0)
    related_signal_ids: list[UUID] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
