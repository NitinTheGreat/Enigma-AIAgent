"""Situation — a long-lived narrative that accumulates evidence over time.

A Situation is NOT an alert.  It is a container for related signals that
together tell a story about something worth tracking.  It carries no
opinions, no risk scores, and no decisions — those belong to later phases.

Lifecycle:  active → dormant → expired
    - active:  receiving evidence recently
    - dormant: no evidence within dormancy window, but still retained
    - expired: no evidence within TTL, eligible for removal
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID

from enigma_reason.domain.signal import Signal
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
        """Compute the current lifecycle state.

        Args:
            dormancy_window: Inactivity duration before a situation goes dormant.
            ttl: Total inactivity duration before a situation expires.

        The dormancy_window MUST be shorter than ttl.
        """
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
