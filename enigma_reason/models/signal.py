"""Pydantic model for incoming signals from the external ML service."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SignalSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncomingSignal(BaseModel):
    """A structured signal received over WebSocket from the ML inference service."""

    signal_id: UUID = Field(..., description="Unique identifier for this signal")
    source: str = Field(..., description="Originating sensor or ML pipeline ID")
    signal_type: str = Field(..., description="Classification label produced by ML")
    severity: SignalSeverity = Field(..., description="Risk severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="ML confidence score")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = Field(default_factory=dict, description="Arbitrary signal metadata")
