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
from typing import Optional
from uuid import UUID

from enigma_reason.domain.signal import Signal
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.foundation.clock import Clock, ClockMode, make_clock
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

    Time domain note:
        Staleness predicates read from an injected Clock and compare against a
        reference timestamp chosen by clock_mode.  See foundation/clock.py and
        paper/EVIDENCE.md section D1.
    """

    __slots__ = (
        "situation_id",
        "created_at",
        "last_updated",
        "version",
        "_evidence",
        "_clock",
        "_clock_mode",
        "_intervals_cache",
    )

    def __init__(
        self,
        situation_id: UUID | None = None,
        clock: Optional[Clock] = None,
        clock_mode: ClockMode | str = ClockMode.SEPARATED,
    ) -> None:
        """
        Args:
            situation_id: Explicit identifier, generated when omitted.
            clock: Time source. Built from clock_mode when omitted.
            clock_mode: Which time domain staleness predicates evaluate in.
        """
        self._clock_mode: ClockMode = ClockMode(clock_mode)
        self._clock: Clock = clock if clock is not None else make_clock(self._clock_mode)
        now = self._clock.now()
        self.situation_id: UUID = situation_id or new_id()
        self.created_at: datetime = now
        self.last_updated: datetime = now
        self.version: int = 1
        self._evidence: list[Signal] = []
        self._intervals_cache: tuple[int, list[float]] | None = None

    @property
    def clock_mode(self) -> ClockMode:
        """Return the time domain this situation evaluates staleness in."""
        return self._clock_mode

    @property
    def clock(self) -> Clock:
        """Return the injected time source."""
        return self._clock

    # ── Mutation ─────────────────────────────────────────────────────────

    def attach_evidence(self, signal: Signal) -> None:
        """Append a signal to this situation's evidence, bump clock and version.

        The clock observes the signal's event timestamp before the ingest
        bookkeeping is updated, so a replay clock reports the newly observed
        instant rather than the previous one.
        """
        self._evidence.append(signal)
        self._clock.observe(signal.timestamp)
        self.last_updated = self._clock.now()
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
        elapsed = self._clock.now() - self.last_updated
        if elapsed > ttl:
            return SituationLifecycle.EXPIRED
        if elapsed > dormancy_window:
            return SituationLifecycle.DORMANT
        return SituationLifecycle.ACTIVE

    def is_expired(self, ttl: timedelta) -> bool:
        """Return True if the situation has received no evidence within *ttl*."""
        return (self._clock.now() - self.last_updated) > ttl

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

        The result is memoised against the version counter, which advances on
        every attach and so invalidates the cache exactly when the evidence
        changes. Section C4 of paper/EVIDENCE.md records that this sort ran at
        least three times per analysis over an evidence list that is never
        trimmed, in a path executed once per arriving signal. Memoising leaves
        the asymptotic cost unchanged but removes the repeats within a single
        analysis, which is where the constant factor was.
        """
        if len(self._evidence) < 2:
            return []
        cached = self._intervals_cache
        if cached is not None and cached[0] == self.version:
            return list(cached[1])
        timestamps = sorted(s.timestamp for s in self._evidence)
        intervals = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        self._intervals_cache = (self.version, intervals)
        return list(intervals)

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

    def _staleness_reference(self) -> Optional[datetime]:
        """Return the timestamp staleness is measured from, per clock mode.

        WALL measures from the ingest bookkeeping timestamp so that both sides
        of the comparison come from the host clock. CONFLATED and SEPARATED both
        measure from the newest event timestamp; they differ in what now() is.

        Returns:
            The reference instant, or None when there is no evidence.
        """
        if not self._evidence:
            return None
        if self._clock_mode is ClockMode.WALL:
            return self.last_updated
        return self.last_seen_at

    def staleness(self) -> timedelta:
        """Return elapsed time between now and the staleness reference.

        Returns:
            A zero duration when there is no evidence to be stale relative to.
        """
        reference = self._staleness_reference()
        if reference is None:
            return timedelta(0)
        return self._clock.now() - reference

    def is_quiet(self, quiet_window: timedelta) -> bool:
        """True if no new events have arrived within the quiet window.

        Args:
            quiet_window: Duration of inactivity that constitutes "quiet".

        This is a pure clock observation — no opinion on what "quiet" means.
        Which clock, and which reference timestamp, is decided by clock_mode.
        """
        if not self._evidence:
            return True
        return self.staleness() > quiet_window

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
        last_event_age = self.staleness().total_seconds()

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

    def summary(
        self,
        dormancy_window: Optional[timedelta] = None,
        ttl: Optional[timedelta] = None,
    ) -> dict:
        """Lightweight summary suitable for acknowledgements and dashboards.

        Contains no decisions or intelligence, only structural facts derived
        from the evidence already attached.

        Args:
            dormancy_window: Supplied to report a lifecycle label. Omitting it
                omits the label rather than guessing a window.
            ttl: Supplied alongside dormancy_window for the same reason.

        Returns:
            A JSON-serialisable dict. The max_anomaly, sources, last_activity
            and lifecycle keys exist because the dashboard reads them; see
            paper/EVIDENCE.md section F1.
        """
        anomaly_scores = [signal.anomaly_score for signal in self._evidence]
        lifecycle = None
        if dormancy_window is not None and ttl is not None:
            lifecycle = self.lifecycle_state(dormancy_window, ttl).value

        return {
            "situation_id": str(self.situation_id),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "last_activity": self.last_updated.isoformat(),
            "lifecycle": lifecycle,
            "version": self.version,
            "evidence_count": self.evidence_count,
            "signal_types": list({s.signal_type.value for s in self._evidence}),
            "entities": list({str(s.entity) for s in self._evidence if s.entity}),
            "sources": list({s.source for s in self._evidence}),
            "abstained_evidence_count": sum(
                1 for signal in self._evidence if signal.abstained
            ),
            "max_anomaly": max(anomaly_scores) if anomaly_scores else 0.0,
            "mean_anomaly": (
                sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
            ),
        }

    # ── Dunder ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Situation(id={self.situation_id!s}, "
            f"v={self.version}, "
            f"evidence={self.evidence_count}, "
            f"age={self._clock.now() - self.created_at})"
        )
