"""SituationTemporalSnapshot — an immutable point-in-time temporal observation.

This is a pure data structure.  It contains no opinions, no thresholds,
and no decisions.  It captures what the clock says about a situation
at the moment it is created.

Future reasoning layers will consume these snapshots as input.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SituationTemporalSnapshot(BaseModel):
    """Immutable temporal observation of a situation at a point in time.

    All fields are derived from evidence timestamps and clock state.
    Nothing here is subjective or configurable.
    """

    situation_id: str
    event_count: int = Field(..., description="Total evidence items")
    active_duration_seconds: float = Field(..., description="Seconds between first and last event")
    event_rate_per_minute: float = Field(..., description="Events per minute over active duration")
    last_event_age_seconds: float = Field(..., description="Seconds since the most recent event")
    mean_interval_seconds: float | None = Field(
        None, description="Mean gap between consecutive events (None if < 2 events)"
    )
    burst_detected: bool = Field(..., description="True if recent events are arriving faster than historical average")
    quiet_detected: bool = Field(..., description="True if no events within the quiet window")

    model_config = {"frozen": True}
