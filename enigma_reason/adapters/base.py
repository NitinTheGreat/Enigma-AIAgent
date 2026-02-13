"""Abstract base for signal adapters.

Signal adapters normalise raw payloads from heterogeneous upstream
sources into the canonical Signal model.

Architectural rules:
    1. Adapters must NOT mutate the incoming payload dict.
    2. adapt() must return a fully valid Signal or raise ValueError.
    3. No adapter may call the SituationStore directly.
    4. No ML logic lives inside an adapter — only field mapping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from enigma_reason.domain.signal import Signal


class SignalAdapter(ABC):
    """Base class for converting raw upstream payloads into canonical Signals."""

    @abstractmethod
    def can_handle(self, raw: dict[str, Any]) -> bool:
        """Return True if this adapter knows how to translate *raw*.

        Must be a fast, non-destructive check (e.g. key presence).
        """
        ...

    @abstractmethod
    def adapt(self, raw: dict[str, Any]) -> Signal:
        """Translate a raw payload dict into a validated Signal.

        The input dict must NOT be mutated.

        Raises:
            ValueError: If the payload cannot be normalised.
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of the upstream source this adapter handles."""
        ...
