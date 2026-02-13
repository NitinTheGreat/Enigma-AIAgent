"""Abstract base for signal adapters.

Signal adapters normalise raw payloads from heterogeneous upstream
sources into the canonical Signal model.  This module defines the
interface only — concrete adapters will be added in future phases.

Architectural intent:
    Each upstream source (SIEM, IDS, custom ML pipeline) may produce
    signals in a different schema.  An adapter translates that schema
    into our canonical Signal without modifying the domain model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from enigma_reason.domain.signal import Signal


class SignalAdapter(ABC):
    """Protocol for converting raw upstream payloads into canonical Signals."""

    @abstractmethod
    def can_handle(self, raw: dict[str, Any]) -> bool:
        """Return True if this adapter knows how to translate *raw*."""
        ...

    @abstractmethod
    def adapt(self, raw: dict[str, Any]) -> Signal:
        """Translate a raw payload dict into a validated Signal.

        Raises:
            ValueError: If the payload cannot be normalised.
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of the upstream source this adapter handles."""
        ...
