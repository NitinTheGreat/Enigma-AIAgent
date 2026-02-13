"""Correlation strategies for grouping signals into situations.

A CorrelationStrategy decides which signals belong to the same narrative.
The store depends on this protocol — swap implementations to change grouping
without touching storage logic.
"""

from __future__ import annotations

from typing import Protocol

from enigma_reason.domain.signal import Signal

# The key type returned by correlation strategies.
CorrelationKey = tuple[str, ...]


class CorrelationStrategy(Protocol):
    """Protocol for signal-to-situation grouping."""

    def get_key(self, signal: Signal) -> CorrelationKey:
        """Return a deterministic key that groups related signals."""
        ...


class DefaultCorrelation:
    """Groups by (signal_type, entity).

    Two signals with the same type and entity are placed in the same
    situation.  Anonymous signals (no entity) are bucketed per type.
    """

    def get_key(self, signal: Signal) -> CorrelationKey:
        entity_str = str(signal.entity) if signal.entity else ""
        return (signal.signal_type.value, entity_str)


class EntityCorrelation:
    """Groups by entity only, regardless of signal type.

    All signals about the same entity (e.g. device:server-prod-01) are
    correlated into one situation even if they have different signal types.
    This captures multi-vector patterns (intrusion + escalation + exfil).

    Anonymous signals (no entity) fall back to per-type grouping.
    """

    def get_key(self, signal: Signal) -> CorrelationKey:
        if signal.entity:
            return (str(signal.entity),)
        return (signal.signal_type.value,)
