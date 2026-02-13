"""Adapter Registry — discovers and selects signal adapters.

The registry holds a list of registered SignalAdapters.  When a raw
payload arrives, it iterates through adapters in registration order
and selects the first one whose can_handle() returns True.

No heuristics.  No guessing.  Fail fast if nothing matches.
"""

from __future__ import annotations

import logging
from typing import Any

from enigma_reason.adapters.base import SignalAdapter
from enigma_reason.domain.signal import Signal

logger = logging.getLogger(__name__)


class AdapterStats:
    """Per-adapter ingestion statistics for observability."""

    __slots__ = ("adapter_name", "accepted_count", "rejected_count")

    def __init__(self, adapter_name: str) -> None:
        self.adapter_name = adapter_name
        self.accepted_count: int = 0
        self.rejected_count: int = 0

    def to_dict(self) -> dict:
        return {
            "adapter_name": self.adapter_name,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
        }


class NoAdapterFoundError(Exception):
    """Raised when no registered adapter can handle a payload."""


class AdaptationError(Exception):
    """Raised when a matched adapter fails to translate the payload."""

    def __init__(self, adapter_name: str, reason: str) -> None:
        self.adapter_name = adapter_name
        self.reason = reason
        super().__init__(f"Adapter '{adapter_name}' failed: {reason}")


class AdapterRegistry:
    """Registry of signal adapters with selection and stats tracking.

    Usage:
        registry = AdapterRegistry()
        registry.register(NetworkAnomalyAdapter())
        registry.register(AuthAnomalyAdapter())

        signal = registry.adapt(raw_payload)
    """

    def __init__(self) -> None:
        self._adapters: list[SignalAdapter] = []
        self._stats: dict[str, AdapterStats] = {}

    def register(self, adapter: SignalAdapter) -> None:
        """Add an adapter to the registry."""
        self._adapters.append(adapter)
        self._stats[adapter.source_name] = AdapterStats(adapter.source_name)
        logger.info("Registered adapter: %s", adapter.source_name)

    def adapt(self, raw: dict[str, Any]) -> Signal:
        """Route a raw payload through the first matching adapter.

        Args:
            raw: The raw payload dict from an upstream source.

        Returns:
            A validated canonical Signal.

        Raises:
            NoAdapterFoundError: If no adapter's can_handle() returns True.
            AdaptationError: If the matched adapter fails to translate.
        """
        for adapter in self._adapters:
            if adapter.can_handle(raw):
                stats = self._stats[adapter.source_name]
                try:
                    signal = adapter.adapt(raw)
                    stats.accepted_count += 1
                    logger.debug(
                        "Adapter '%s' accepted payload → signal %s",
                        adapter.source_name,
                        signal.signal_id,
                    )
                    return signal
                except (ValueError, Exception) as exc:
                    stats.rejected_count += 1
                    logger.warning(
                        "Adapter '%s' rejected payload: %s",
                        adapter.source_name,
                        exc,
                    )
                    raise AdaptationError(adapter.source_name, str(exc)) from exc

        raise NoAdapterFoundError(
            f"No adapter can handle payload with keys: {sorted(raw.keys())}"
        )

    @property
    def adapter_names(self) -> list[str]:
        """List of registered adapter names in registration order."""
        return [a.source_name for a in self._adapters]

    @property
    def stats(self) -> list[dict]:
        """Per-adapter stats for observability endpoints."""
        return [s.to_dict() for s in self._stats.values()]

    @property
    def total_accepted(self) -> int:
        return sum(s.accepted_count for s in self._stats.values())

    @property
    def total_rejected(self) -> int:
        return sum(s.rejected_count for s in self._stats.values())
