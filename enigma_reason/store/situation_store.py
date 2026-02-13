"""In-memory Situation store with async-safe access and TTL-based expiry.

Design notes:
    - An asyncio.Lock guards all mutations so concurrent WebSocket handlers
      never corrupt state.
    - Correlation logic is injected via a CorrelationStrategy, defaulting to
      (signal_type, entity) grouping.
    - Situations follow a lifecycle: active → dormant → expired.
    - Dormant situations are retained and reactivated on new evidence.
    - The store does NOT decide *what* a situation means.  It only tracks
      which signals belong together and when to forget them.
    - Phase 2: temporal_summary() exposes aggregate temporal observations.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from uuid import UUID

from enigma_reason.domain.signal import Signal
from enigma_reason.domain.situation import Situation, SituationLifecycle
from enigma_reason.store.correlation import (
    CorrelationKey,
    CorrelationStrategy,
    DefaultCorrelation,
)

logger = logging.getLogger(__name__)


class TemporalSummary:
    """Aggregate temporal observations across all active situations.

    This is an observability object, not a control mechanism.
    """

    __slots__ = (
        "total_situations",
        "active_situations",
        "dormant_situations",
        "bursting_situations",
        "quiet_situations",
        "max_event_rate",
    )

    def __init__(
        self,
        total_situations: int = 0,
        active_situations: int = 0,
        dormant_situations: int = 0,
        bursting_situations: int = 0,
        quiet_situations: int = 0,
        max_event_rate: float = 0.0,
    ) -> None:
        self.total_situations = total_situations
        self.active_situations = active_situations
        self.dormant_situations = dormant_situations
        self.bursting_situations = bursting_situations
        self.quiet_situations = quiet_situations
        self.max_event_rate = max_event_rate

    def to_dict(self) -> dict:
        return {
            "total_situations": self.total_situations,
            "active_situations": self.active_situations,
            "dormant_situations": self.dormant_situations,
            "bursting_situations": self.bursting_situations,
            "quiet_situations": self.quiet_situations,
            "max_event_rate": round(self.max_event_rate, 4),
        }


class SituationStore:
    """Async-safe, in-memory store for active Situations.

    Args:
        ttl: How long a situation may remain without new evidence before
             it is considered expired and eligible for removal.
        dormancy_window: Inactivity period after which a situation is
             considered dormant (but still retained).
        correlation: Strategy for grouping signals into situations.
        burst_factor: Multiplier for burst detection threshold.
        burst_recent_count: Number of recent intervals to evaluate for bursts.
        quiet_window: Duration of inactivity that qualifies as "quiet".
    """

    def __init__(
        self,
        ttl: timedelta = timedelta(minutes=30),
        dormancy_window: timedelta = timedelta(minutes=10),
        correlation: CorrelationStrategy | None = None,
        burst_factor: float = 3.0,
        burst_recent_count: int = 3,
        quiet_window: timedelta = timedelta(minutes=5),
    ) -> None:
        if dormancy_window >= ttl:
            raise ValueError("dormancy_window must be shorter than ttl")

        self._ttl = ttl
        self._dormancy_window = dormancy_window
        self._correlation = correlation or DefaultCorrelation()
        self._burst_factor = burst_factor
        self._burst_recent_count = burst_recent_count
        self._quiet_window = quiet_window
        self._lock = asyncio.Lock()
        self._situations: dict[UUID, Situation] = {}
        self._key_index: dict[CorrelationKey, UUID] = {}

    # ── Public API ───────────────────────────────────────────────────────

    async def ingest(self, signal: Signal) -> Situation:
        """Find-or-create a Situation for *signal*, attach evidence, return it.

        This is the single entry point used by the WebSocket handler.
        Dormant situations are reactivated when new evidence arrives.
        """
        async with self._lock:
            situation = self._find_or_create(signal)
            situation.attach_evidence(signal)
            logger.debug(
                "Ingested signal %s → situation %s (v=%d, evidence=%d)",
                signal.signal_id,
                situation.situation_id,
                situation.version,
                situation.evidence_count,
            )
            return situation

    async def get(self, situation_id: UUID) -> Situation | None:
        """Retrieve a situation by ID, or None if not found / expired."""
        async with self._lock:
            return self._situations.get(situation_id)

    async def expire_stale(self) -> list[UUID]:
        """Remove all situations that have exceeded the TTL.

        Returns the IDs of expired situations for logging / diagnostics.
        Dormant situations are NOT removed — only truly expired ones.
        """
        async with self._lock:
            expired_ids: list[UUID] = [
                sid
                for sid, sit in self._situations.items()
                if sit.is_expired(self._ttl)
            ]
            for sid in expired_ids:
                self._remove(sid)
            if expired_ids:
                logger.info("Expired %d stale situation(s)", len(expired_ids))
            return expired_ids

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._situations)

    async def dormant_count(self) -> int:
        """Count situations currently in dormant lifecycle state."""
        async with self._lock:
            return sum(
                1 for sit in self._situations.values()
                if sit.lifecycle_state(self._dormancy_window, self._ttl)
                == SituationLifecycle.DORMANT
            )

    # ── Temporal Observations (Phase 2) ──────────────────────────────────

    async def temporal_summary(self) -> TemporalSummary:
        """Aggregate temporal observations across all situations.

        This is observability, not control.  It mutates nothing.
        """
        async with self._lock:
            total = len(self._situations)
            active = 0
            dormant = 0
            bursting = 0
            quiet = 0
            max_rate = 0.0

            for sit in self._situations.values():
                state = sit.lifecycle_state(self._dormancy_window, self._ttl)
                if state == SituationLifecycle.ACTIVE:
                    active += 1
                elif state == SituationLifecycle.DORMANT:
                    dormant += 1

                if sit.is_bursting(self._burst_factor, self._burst_recent_count):
                    bursting += 1
                if sit.is_quiet(self._quiet_window):
                    quiet += 1

                rate = sit.event_rate
                if rate > max_rate:
                    max_rate = rate

            return TemporalSummary(
                total_situations=total,
                active_situations=active,
                dormant_situations=dormant,
                bursting_situations=bursting,
                quiet_situations=quiet,
                max_event_rate=max_rate,
            )

    # ── Internals ────────────────────────────────────────────────────────

    def _find_or_create(self, signal: Signal) -> Situation:
        """Must be called while holding self._lock."""
        key = self._correlation.get_key(signal)
        sid = self._key_index.get(key)

        if sid and sid in self._situations:
            existing = self._situations[sid]
            # If the existing situation expired, discard it and start fresh
            if existing.is_expired(self._ttl):
                self._remove(sid)
            else:
                # Reactivate dormant situations — new evidence wakes them up
                return existing

        # Create a new situation
        situation = Situation()
        self._situations[situation.situation_id] = situation
        self._key_index[key] = situation.situation_id
        logger.info("Created situation %s for key %s", situation.situation_id, key)
        return situation

    def _remove(self, situation_id: UUID) -> None:
        """Must be called while holding self._lock."""
        self._situations.pop(situation_id, None)
        # Clean up key index
        self._key_index = {
            k: v for k, v in self._key_index.items() if v != situation_id
        }
