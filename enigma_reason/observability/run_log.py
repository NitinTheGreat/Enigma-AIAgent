"""
Module: enigma_reason/observability/run_log.py

One JSONL record per reasoning iteration.

Granularity is per iteration rather than per analysis because Level 7 scores
outcomes that only exist between iterations: whether a leader was declared
before the evidence justified it, and how many iterations passed before the
loop terminated. An analysis level record cannot express either.

Hypothesis text is stored as a short digest rather than verbatim. The digest
keeps a long run small while still revealing when the language model
regenerates a hypothesis it has already proposed, which is the only property
of the text the analysis needs.

The writer is deliberately defensive. A logging failure that propagated into
the reasoning path would turn an observability feature into an availability
risk, so every write is wrapped and failures are counted rather than raised.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol
from uuid import uuid4

from enigma_reason.domain.hypothesis import UNKNOWN_HYPOTHESIS_ID

logger = logging.getLogger(__name__)

TEXT_HASH_LENGTH = 12


def text_hash(text: str) -> str:
    """Return a short stable digest of hypothesis text.

    Args:
        text: The hypothesis description as generated.

    Returns:
        The leading hex characters of the SHA256 digest of the stripped,
        case folded text, so that trivial whitespace or capitalisation
        differences do not register as a new hypothesis.
    """
    normalised = " ".join(text.strip().casefold().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:TEXT_HASH_LENGTH]


class RecordSink(Protocol):
    """Anything that accepts finished iteration records."""

    def write(self, record: dict[str, Any]) -> None:
        """Persist one record."""


class RunLogWriter:
    """Append only JSONL sink, safe to share across threads.

    Args:
        path: Destination file. Parent directories are created on demand.

    Attributes:
        written: Records successfully persisted.
        dropped: Records lost to an exception, which is the number the caller
            should report rather than assume is zero.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.written = 0
        self.dropped = 0
        self._lock = threading.Lock()
        self._handle = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        except Exception as exc:
            logger.error("Run log could not be opened at %s: %s", self.path, exc)

    def write(self, record: dict[str, Any]) -> None:
        """Append one record, counting rather than raising on failure."""
        if self._handle is None:
            self.dropped += 1
            return
        try:
            line = json.dumps(record, separators=(",", ":"), default=str)
            with self._lock:
                self._handle.write(line + "\n")
                self.written += 1
        except Exception as exc:
            self.dropped += 1
            logger.error("Run log record dropped: %s", exc)

    def flush(self) -> None:
        """Push buffered records to the operating system."""
        if self._handle is None:
            return
        try:
            with self._lock:
                self._handle.flush()
        except Exception as exc:
            logger.error("Run log flush failed: %s", exc)

    def close(self) -> None:
        """Flush and release the file handle."""
        if self._handle is None:
            return
        try:
            with self._lock:
                self._handle.flush()
                self._handle.close()
        except Exception as exc:
            logger.error("Run log close failed: %s", exc)
        finally:
            self._handle = None

    def __enter__(self) -> "RunLogWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class ListSink:
    """In memory sink used by tests and by the offline replay summary."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.written = 0
        self.dropped = 0

    def write(self, record: dict[str, Any]) -> None:
        """Retain one record."""
        self.records.append(record)
        self.written += 1


def _hypothesis_view(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Project one hypothesis onto the fields the run log carries."""
    hypothesis_id = hypothesis.get("hypothesis_id")
    return {
        "hypothesis_id": hypothesis_id,
        "text_hash": text_hash(str(hypothesis.get("description", ""))),
        "confidence": hypothesis.get("confidence"),
        "confidence_previous": hypothesis.get("confidence_previous"),
        "confidence_before_inertia": hypothesis.get("confidence_before_inertia"),
        "inertia_clamped": bool(hypothesis.get("inertia_clamped", False)),
        "is_unknown": hypothesis_id == UNKNOWN_HYPOTHESIS_ID,
        "status": hypothesis.get("status"),
    }


def _dominant(hypotheses: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest confidence hypothesis that is still in play."""
    live = [h for h in hypotheses if h.get("status") in ("active", "converged")]
    if not live:
        return None
    return max(live, key=lambda h: h.get("confidence", 0.0))


class IterationRecorder:
    """Turns a stream of graph states into one record per reasoning iteration.

    The recorder is fed the full state after each LangGraph super step. Only
    update_convergence advances iteration_count, so a rise in that field marks
    the boundary of a completed iteration and is the trigger for a record.

    Termination is not knowable until the loop exits, so the final record is
    held back until finalise is called and is stamped with the reason the loop
    stopped.

    Args:
        sink: Where finished records go.
        run_id: Identifier tying every record of this run together. Generated
            when omitted.
        situation_id: The situation under analysis.
        clock_mode: Which time domain the situation evaluates staleness in.
        event_timestamp: Newest event timestamp known to the situation, which
            is distinct from wall time under replay and is the whole point of
            recording both.
    """

    def __init__(
        self,
        sink: RecordSink,
        *,
        run_id: str | None = None,
        situation_id: str = "",
        clock_mode: str = "",
        event_timestamp: datetime | None = None,
    ) -> None:
        self.sink = sink
        self.run_id = run_id or str(uuid4())
        self.situation_id = situation_id
        self.clock_mode = clock_mode
        self.event_timestamp = event_timestamp
        self.last_iteration = 0
        self.emitted = 0
        self._pending: dict[str, Any] | None = None

    def observe(self, state: dict[str, Any]) -> None:
        """Consider one graph state, emitting a record if an iteration closed."""
        try:
            iteration = int(state.get("iteration_count", 0))
        except Exception:
            return
        if iteration <= self.last_iteration:
            return
        self.last_iteration = iteration
        try:
            record = self._build(state, iteration)
        except Exception as exc:
            logger.error("Iteration record could not be built: %s", exc)
            return
        self._flush_pending()
        self._pending = record

    def finalise(self, final_state: dict[str, Any]) -> None:
        """Stamp the held back record with why the loop stopped, then emit it."""
        if self._pending is None:
            return
        try:
            self._pending["terminated"] = True
            self._pending["termination_reason"] = self._termination_reason(final_state)
        except Exception as exc:
            logger.error("Termination reason could not be derived: %s", exc)
        self._flush_pending()

    def _flush_pending(self) -> None:
        """Send the buffered record onward."""
        if self._pending is None:
            return
        record, self._pending = self._pending, None
        try:
            self.sink.write(record)
            self.emitted += 1
        except Exception as exc:
            logger.error("Iteration record dropped by sink: %s", exc)

    @staticmethod
    def _termination_reason(final_state: dict[str, Any]) -> str:
        """Name the exit condition, matching the order check_convergence tests."""
        convergence = float(final_state.get("convergence_score", 0.0))
        threshold = float(final_state.get("convergence_threshold", 0.8))
        iteration = int(final_state.get("iteration_count", 0))
        max_iterations = int(final_state.get("max_iterations", 3))
        if convergence >= threshold:
            return "converged"
        if iteration >= max_iterations:
            return "max_iterations"
        return "unspecified"

    def _build(self, state: dict[str, Any], iteration: int) -> dict[str, Any]:
        """Assemble one record from a completed iteration's state."""
        reasoning = state.get("reasoning_snapshot", {}) or {}
        hypotheses = state.get("hypotheses", []) or []
        leader = _dominant(hypotheses)

        return {
            "run_id": self.run_id,
            "situation_id": self.situation_id or state.get("situation_id", ""),
            "iteration": iteration,
            "wall_timestamp": datetime.now(timezone.utc).isoformat(),
            "event_timestamp": (
                self.event_timestamp.isoformat() if self.event_timestamp else None
            ),
            "clock_mode": self.clock_mode,
            "trend": reasoning.get("trend"),
            "convergence_score": state.get("convergence_score"),
            "evidence_count": reasoning.get("evidence_count"),
            "source_diversity": reasoning.get("source_diversity"),
            "mean_anomaly": reasoning.get("mean_anomaly_score"),
            "burst_detected": reasoning.get("burst_detected"),
            "is_quiet": reasoning.get("quiet_detected"),
            "abstained_evidence_count": reasoning.get("abstained_evidence_count"),
            "abstention_fraction": reasoning.get("abstained_fraction"),
            "hypotheses": [_hypothesis_view(h) for h in hypotheses],
            "dominant_hypothesis_id": leader.get("hypothesis_id") if leader else None,
            "dominant_iterations": leader.get("dominant_iterations", 0) if leader else 0,
            "belief_stability_score": state.get("belief_stability_score"),
            "undecided_iterations": state.get("undecided_iterations"),
            "terminated": False,
            "termination_reason": None,
        }
