"""
Module: enigma_reason/foundation/clock.py

Timezone-aware clock abstraction and the three clock modes the reasoning stack
supports.

All timestamps in enigma-reason MUST be UTC-aware. This module is the single
source of "now" so tests can monkey-patch it trivially.

Two distinct time domains exist in this system and conflating them is the D1
defect recorded in paper/EVIDENCE.md section D1:

    ingest time   when the collector received a signal, read from the host clock
    event time    when the sensor observed the flow, carried on the signal itself

Replaying archival capture makes these differ by years. UNSW-NB15 spans
2015-01-22 to 2015-02-18, so a wall-clock staleness predicate evaluated against
an event timestamp returns roughly eleven and a half years against a five minute
window, and therefore reports every situation as quiet on every replay.

ClockMode selects which domain the staleness predicates read:

    CONFLATED   host clock compared against event timestamps, the original defect
    WALL        host clock compared against ingest timestamps, internally consistent
    SEPARATED   event clock compared against event timestamps, internally consistent

CONFLATED is retained deliberately. Level 8 runs all three as an experiment, so
the defect is parameterised rather than deleted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Protocol, runtime_checkable


class ClockMode(str, Enum):
    """Selects which time domain staleness predicates are evaluated in."""

    CONFLATED = "conflated"
    WALL = "wall"
    SEPARATED = "separated"


def utc_now() -> datetime:
    """Return the current host time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


@runtime_checkable
class Clock(Protocol):
    """Contract for a source of "now" within a single time domain."""

    def now(self) -> datetime:
        """Return the current instant in this clock's time domain."""
        ...

    def observe(self, timestamp: datetime) -> None:
        """Inform the clock of an observed event timestamp."""
        ...


class WallClock:
    """Host clock. Ignores observed event timestamps entirely."""

    __slots__ = ()

    def now(self) -> datetime:
        """Return the current host time."""
        return utc_now()

    def observe(self, timestamp: datetime) -> None:
        """Discard the observation. The host clock advances on its own."""
        return None


class ReplayClock:
    """
    Event clock. Advances monotonically to the latest observed event timestamp.

    Before the first observation there is no event time to report, so the host
    clock is used as a seed. This only affects a situation constructed but never
    given evidence, which has no temporal metrics to compute anyway.
    """

    __slots__ = ("_latest",)

    def __init__(self, latest: Optional[datetime] = None) -> None:
        """
        Args:
            latest: Optional starting instant, used when rehydrating a clock.
        """
        self._latest: Optional[datetime] = latest

    @property
    def latest_observed(self) -> Optional[datetime]:
        """Return the newest observed event timestamp, or None before any."""
        return self._latest

    def now(self) -> datetime:
        """Return the newest observed event timestamp, falling back to host time."""
        if self._latest is None:
            return utc_now()
        return self._latest

    def observe(self, timestamp: datetime) -> None:
        """Advance to timestamp when it is newer than everything seen so far."""
        if timestamp is None:
            return None
        if self._latest is None or timestamp > self._latest:
            self._latest = timestamp
        return None


def make_clock(mode: ClockMode | str) -> Clock:
    """
    Build the clock implementation matching a mode.

    Args:
        mode: A ClockMode, or its string value.

    Returns:
        A ReplayClock for SEPARATED, a WallClock for CONFLATED and WALL. The
        difference between the latter two is not the clock but which reference
        timestamp the caller compares against, which Situation decides.

    Raises:
        ValueError: If mode is not a recognised ClockMode value.
    """
    resolved = ClockMode(mode)
    if resolved is ClockMode.SEPARATED:
        return ReplayClock()
    return WallClock()
