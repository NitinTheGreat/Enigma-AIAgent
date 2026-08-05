"""
Module: enigma_reason/observability/latency.py

Per stage timing for the six boundaries an arriving signal crosses.

The stages are reported separately and never summed into a single pipeline
number. The language model call is expected to dominate the other five by one
to two orders of magnitude, and an aggregate would hide exactly the fact the
paper needs to state: the reasoning layer is fast and the model it calls is
not.

Timings are held as raw samples rather than a streaming digest. A Level 9
suite produces a few thousand samples per stage, which is small enough that
exact percentiles cost less than the error bars of an approximation.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """The six instrumented boundaries, in the order a signal crosses them."""

    INGEST = "signal_ingest"
    ATTACH = "situation_attach"
    DETERMINISTIC = "deterministic_reasoning"
    LANGGRAPH = "langgraph_gemini"
    EXPLANATION = "explanation_build"
    BROADCAST = "dashboard_broadcast"


def percentile(samples: list[float], fraction: float) -> float:
    """Return the linearly interpolated percentile of a sample list.

    Args:
        samples: Unsorted observations. Not mutated.
        fraction: Position in the unit interval, so 0.95 is the 95th
            percentile.

    Returns:
        The interpolated value, or 0.0 when there are no observations.
    """
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class LatencyRecorder:
    """Collects per stage durations in milliseconds.

    Safe to call from both the event loop and the thread pool, because the
    LangGraph stage runs under asyncio.to_thread while the others do not.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def record(self, stage: Stage | str, milliseconds: float) -> None:
        """Add one observation, swallowing any failure."""
        try:
            name = stage.value if isinstance(stage, Stage) else str(stage)
            with self._lock:
                self._samples.setdefault(name, []).append(milliseconds)
        except Exception as exc:
            logger.error("Latency sample dropped: %s", exc)

    @contextmanager
    def measure(self, stage: Stage | str) -> Iterator[None]:
        """Time the enclosed block and record it even when the block raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - started) * 1000.0)

    def count(self, stage: Stage | str) -> int:
        """Return how many observations a stage has."""
        name = stage.value if isinstance(stage, Stage) else str(stage)
        with self._lock:
            return len(self._samples.get(name, []))

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return count, mean and the p50, p95 and p99 percentiles per stage.

        Stages appear in pipeline order so the report reads as the path a
        signal actually takes, with any unrecognised stage name appended.
        """
        with self._lock:
            snapshot = {name: list(values) for name, values in self._samples.items()}

        ordered_names = [s.value for s in Stage if s.value in snapshot]
        ordered_names += [n for n in snapshot if n not in ordered_names]

        result: dict[str, dict[str, Any]] = {}
        for name in ordered_names:
            values = snapshot[name]
            result[name] = {
                "count": len(values),
                "mean_ms": round(sum(values) / len(values), 3) if values else 0.0,
                "p50_ms": round(percentile(values, 0.50), 3),
                "p95_ms": round(percentile(values, 0.95), 3),
                "p99_ms": round(percentile(values, 0.99), 3),
                "max_ms": round(max(values), 3) if values else 0.0,
            }
        return result

    def reset(self) -> None:
        """Discard all observations."""
        with self._lock:
            self._samples.clear()
