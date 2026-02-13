"""Situation — a long-lived narrative that accumulates evidence over time.

A Situation is NOT an alert.  It is a container for related signals that
together tell a story about something worth tracking.  It carries no
opinions, no risk scores, and no decisions — those belong to later phases.

Lifecycle:  active → dormant → expired
    - active:  receiving evidence recently
    - dormant: no evidence within dormancy window, but still retained
    - expired: no evidence within TTL, eligible for removal

Temporal awareness (Phase 2):
    Situations expose read-only temporal metrics derived from evidence
    timestamps.  These are observations, not conclusions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID

from enigma_reason.domain.signal import Signal
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.foundation.clock import utc_now
from enigma_reason.foundation.identifiers import new_id


class SituationLifecycle(str, Enum):
    """Explicit lifecycle states for a situation."""

    ACTIVE = "active"
    DORMANT = "dormant"
    EXPIRED = "expired"


class Situation:
    """A mutable, long-lived evidence container.

    Thread-safety note:
        Individual Situation objects are mutated *only* while the caller
        holds the SituationStore lock.  They are not themselves locked.
    """

    __slots__ = ("situation_id", "created_at", "last_updated", "version", "_evidence")

    def __init__(self, situation_id: UUID | None = None) -> None:
        now = utc_now()
        self.situation_id: UUID = situation_id or new_id()
        self.created_at: datetime = now
        self.last_updated: datetime = now
        self.version: int = 1
        self._evidence: list[Signal] = []

    # ── Mutation ─────────────────────────────────────────────────────────

    def attach_evidence(self, signal: Signal) -> None:
        """Append a signal to this situation's evidence, bump clock and version."""
        self._evidence.append(signal)
        self.last_updated = utc_now()
        self.version += 1

    # ── Queries ──────────────────────────────────────────────────────────

    @property
    def evidence(self) -> list[Signal]:
        """Read-only view of accumulated evidence."""
        return list(self._evidence)

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

    def lifecycle_state(
        self,
        dormancy_window: timedelta,
        ttl: timedelta,
    ) -> SituationLifecycle:
        """Compute the current lifecycle state."""
        elapsed = utc_now() - self.last_updated
        if elapsed > ttl:
            return SituationLifecycle.EXPIRED
        if elapsed > dormancy_window:
            return SituationLifecycle.DORMANT
        return SituationLifecycle.ACTIVE

    def is_expired(self, ttl: timedelta) -> bool:
        """Return True if the situation has received no evidence within *ttl*."""
        return (utc_now() - self.last_updated) > ttl

    def is_dormant(self, dormancy_window: timedelta, ttl: timedelta) -> bool:
        """Return True if dormant (inactive past window but not yet expired)."""
        return self.lifecycle_state(dormancy_window, ttl) == SituationLifecycle.DORMANT

    # ── Temporal Metrics (Phase 2) ───────────────────────────────────────

    @property
    def first_seen_at(self) -> datetime | None:
        """Timestamp of the earliest piece of evidence."""
        if not self._evidence:
            return None
        return min(s.timestamp for s in self._evidence)

    @property
    def last_seen_at(self) -> datetime | None:
        """Timestamp of the most recent piece of evidence."""
        if not self._evidence:
            return None
        return max(s.timestamp for s in self._evidence)

    @property
    def active_duration(self) -> float:
        """Seconds between the first and last evidence timestamp.

        Returns 0.0 if fewer than 2 events.
        """
        first, last = self.first_seen_at, self.last_seen_at
        if first is None or last is None or first == last:
            return 0.0
        return (last - first).total_seconds()

    @property
    def event_intervals(self) -> list[float]:
        """Time gaps (seconds) between consecutive evidence events.

        Events are sorted by their signal timestamp.
        Returns an empty list if fewer than 2 events.
        """
        if len(self._evidence) < 2:
            return []
        timestamps = sorted(s.timestamp for s in self._evidence)
        return [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]

    @property
    def event_rate(self) -> float:
        """Events per minute over the active duration.

        Returns 0.0 if duration is zero (single event or no events).
        """
        dur = self.active_duration
        if dur == 0.0:
            return 0.0
        return (self.evidence_count / dur) * 60.0

    def is_bursting(
        self,
        burst_factor: float = 3.0,
        recent_count: int = 3,
    ) -> bool:
        """True if recent events are arriving significantly faster than average.

        Compares the mean interval of the last *recent_count* events against
        the overall mean interval, scaled by *burst_factor*.

        Args:
            burst_factor: How many times faster than average qualifies as burst.
            recent_count: Number of recent intervals to compare.

        Returns False if there is insufficient data (< recent_count + 1 events).
        """
        intervals = self.event_intervals
        if len(intervals) < recent_count:
            return False

        overall_mean = sum(intervals) / len(intervals)
        if overall_mean == 0.0:
            return False

        recent_intervals = intervals[-recent_count:]
        recent_mean = sum(recent_intervals) / len(recent_intervals)

        # Burst = recent intervals are burst_factor times shorter than average
        return recent_mean < (overall_mean / burst_factor)

    def is_quiet(self, quiet_window: timedelta) -> bool:
        """True if no new events have arrived within the quiet window.

        Args:
            quiet_window: Duration of inactivity that constitutes "quiet".

        This is a pure clock observation — no opinion on what "quiet" means.
        """
        if not self._evidence:
            return True
        last = self.last_seen_at
        assert last is not None
        return (utc_now() - last) > quiet_window

    def temporal_snapshot(
        self,
        burst_factor: float = 3.0,
        recent_count: int = 3,
        quiet_window: timedelta = timedelta(minutes=5),
    ) -> SituationTemporalSnapshot:
        """Create an immutable temporal snapshot of this situation's current state."""
        intervals = self.event_intervals
        mean_interval = (
            sum(intervals) / len(intervals) if intervals else None
        )
        last_seen = self.last_seen_at
        last_event_age = (
            (utc_now() - last_seen).total_seconds() if last_seen else 0.0
        )

        return SituationTemporalSnapshot(
            situation_id=str(self.situation_id),
            event_count=self.evidence_count,
            active_duration_seconds=self.active_duration,
            event_rate_per_minute=self.event_rate,
            last_event_age_seconds=last_event_age,
            mean_interval_seconds=mean_interval,
            burst_detected=self.is_bursting(burst_factor, recent_count),
            quiet_detected=self.is_quiet(quiet_window),
        )

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Lightweight summary suitable for acknowledgements and logging.

        Contains no decisions, scores, or intelligence — just structural facts.
        """
        return {
            "situation_id": str(self.situation_id),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "version": self.version,
            "evidence_count": self.evidence_count,
            "signal_types": list({s.signal_type.value for s in self._evidence}),
            "entities": list({str(s.entity) for s in self._evidence if s.entity}),
        }

    # ── Dunder ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Situation(id={self.situation_id!s}, "
            f"v={self.version}, "
            f"evidence={self.evidence_count}, "
            f"age={utc_now() - self.created_at})"
        )
