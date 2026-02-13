"""Situation — a long-lived narrative that accumulates evidence over time.

A Situation is NOT an alert.  It is a container for related signals that
together tell a story about something worth tracking.  It carries no
opinions, no risk scores, and no decisions — those belong to later phases.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from enigma_reason.domain.signal import Signal
from enigma_reason.foundation.clock import utc_now
from enigma_reason.foundation.identifiers import new_id


class Situation:
    """A mutable, long-lived evidence container.

    Thread-safety note:
        Individual Situation objects are mutated *only* while the caller
        holds the SituationStore lock.  They are not themselves locked.
    """

    __slots__ = ("situation_id", "created_at", "last_updated", "_evidence")

    def __init__(self, situation_id: UUID | None = None) -> None:
        now = utc_now()
        self.situation_id: UUID = situation_id or new_id()
        self.created_at: datetime = now
        self.last_updated: datetime = now
        self._evidence: list[Signal] = []

    # ── Mutation ─────────────────────────────────────────────────────────

    def attach_evidence(self, signal: Signal) -> None:
        """Append a signal to this situation's evidence and bump the clock."""
        self._evidence.append(signal)
        self.last_updated = utc_now()

    # ── Queries ──────────────────────────────────────────────────────────

    @property
    def evidence(self) -> list[Signal]:
        """Read-only view of accumulated evidence."""
        return list(self._evidence)

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

    def is_expired(self, ttl: timedelta) -> bool:
        """Return True if the situation has received no evidence within *ttl*."""
        return (utc_now() - self.last_updated) > ttl

    def summary(self) -> dict:
        """Lightweight summary suitable for acknowledgements and logging.

        Contains no decisions, scores, or intelligence — just structural facts.
        """
        return {
            "situation_id": str(self.situation_id),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "evidence_count": self.evidence_count,
            "signal_types": list({s.signal_type.value for s in self._evidence}),
            "entities": list({str(s.entity) for s in self._evidence if s.entity}),
        }

    # ── Dunder ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Situation(id={self.situation_id!s}, "
            f"evidence={self.evidence_count}, "
            f"age={utc_now() - self.created_at})"
        )
