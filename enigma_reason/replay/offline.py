"""
Module: enigma_reason/replay/offline.py

Runs the reasoning stack over a fixed list of signals, in process.

The live path reaches the reasoning graph through a websocket handler, an
async store, a background task and a broadcast. None of that is part of what
Levels 8 and 9 measure, and all of it makes a run slow and non deterministic.
This module keeps the store, the deterministic reasoning engine, the graph and
the explanation builder, and drops the transport.

Two model backends are supported. The mock returns a deterministic response
derived from the prompt, which makes a hundred situation replay finish in
under a second and lets the run logger be tested without spending money. The
real backend is the ordinary Gemini factory, optionally behind the response
cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from uuid import UUID, uuid4

from enigma_reason.core.reasoning_engine import ReasoningEngine
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import EntityRef, Signal
from enigma_reason.domain.situation import Situation
from enigma_reason.explain.builder import build_explanation
from enigma_reason.foundation.clock import ClockMode
from enigma_reason.graph.builder import EpistemicControls
from enigma_reason.graph.runner import run_reasoning
from enigma_reason.observability.latency import LatencyRecorder, Stage
from enigma_reason.observability.run_log import RecordSink
from enigma_reason.store.correlation import CorrelationStrategy
from enigma_reason.store.situation_store import SituationStore

logger = logging.getLogger(__name__)

MOCK_BENIGN_DESCRIPTION = "Routine automated activity consistent with baseline"
MOCK_ANOMALOUS_DESCRIPTION = "Coordinated probing across multiple source vectors"
MOCK_TRANSFER_DESCRIPTION = "Elevated transfer volume outside the usual window"


class MockLLM:
    """Deterministic stand in for the language model.

    The response depends only on the prompt, so a replay is reproducible and
    two configurations that assemble the same prompt receive the same
    hypotheses. Confidences are drawn from the prompt digest rather than fixed,
    so the downstream evaluation and convergence logic sees a realistic spread
    instead of a constant.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.calls = 0

    def invoke(self, prompt: str) -> "MockResponse":
        """Return three hypotheses derived deterministically from the prompt."""
        self.calls += 1
        digest = hashlib.sha256(f"{self.seed}:{prompt}".encode("utf-8")).digest()
        spread = [0.1 + (digest[i] / 255.0) * 0.4 for i in range(3)]
        payload = [
            {
                "description": MOCK_BENIGN_DESCRIPTION,
                "confidence": round(spread[0], 3),
                "is_benign": True,
            },
            {
                "description": MOCK_ANOMALOUS_DESCRIPTION,
                "confidence": round(spread[1], 3),
                "is_benign": False,
            },
            {
                "description": MOCK_TRANSFER_DESCRIPTION,
                "confidence": round(spread[2], 3),
                "is_benign": False,
            },
        ]
        return MockResponse(content=json.dumps(payload))


@dataclass(frozen=True)
class MockResponse:
    """Minimal message object exposing the attribute the nodes read."""

    content: str


def mock_llm_factory(seed: int = 42) -> Callable[[], MockLLM]:
    """Return a factory producing a shared deterministic mock model."""
    model = MockLLM(seed=seed)

    def factory() -> MockLLM:
        return model

    return factory


def load_signals(path: str | Path, limit: int | None = None) -> list[Signal]:
    """Read signals from a JSONL file produced by the sensor export.

    Rows that cannot be turned into a Signal are skipped and counted in the
    log rather than aborting a long replay.

    Args:
        path: JSONL file, one signal per line.
        limit: Stop after this many successfully parsed signals.

    Returns:
        The parsed signals in file order.
    """
    source = Path(path)
    signals: list[Signal] = []
    skipped = 0

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                signals.append(_signal_from_row(json.loads(line)))
            except Exception as exc:
                skipped += 1
                logger.debug("Skipping unparseable replay row: %s", exc)
                continue
            if limit is not None and len(signals) >= limit:
                break

    if skipped:
        logger.warning("Skipped %d unparseable rows in %s", skipped, source)
    return signals


def _signal_from_row(row: dict[str, Any]) -> Signal:
    """Build a Signal from one exported sensor row."""
    entity_identifier = row.get("entity_device")
    entity = (
        EntityRef(kind=EntityKind.DEVICE, identifier=str(entity_identifier))
        if entity_identifier
        else None
    )
    raw_type = str(row.get("emitted_signal_type", "unknown")).strip().lower()
    try:
        signal_type = SignalType(raw_type)
    except ValueError:
        signal_type = SignalType.UNKNOWN

    return Signal(
        signal_id=UUID(row["signal_id"]) if row.get("signal_id") else uuid4(),
        timestamp=row.get("timestamp") or datetime.now(timezone.utc),
        signal_type=signal_type,
        entity=entity,
        anomaly_score=float(row.get("anomaly_score", 0.0)),
        confidence=float(row.get("calibrated_confidence", row.get("confidence", 0.5))),
        predicted_class_confidence=row.get("predicted_class_confidence"),
        predictive_entropy=row.get("predictive_entropy"),
        source=str(row.get("source", "offline-replay")),
        abstained=bool(row.get("abstained", False)),
        calibrated_confidence=row.get("calibrated_confidence"),
    )


def synthetic_signals(
    count: int,
    *,
    seed: int = 42,
    devices: int = 8,
    start: datetime | None = None,
    interval_seconds: float = 15.0,
) -> list[Signal]:
    """Generate a deterministic signal stream for tests and smoke runs.

    Args:
        count: How many signals to produce.
        seed: Drives the deterministic score and type sequence.
        devices: How many distinct entities to spread the signals across, which
            controls how many situations the store will create.
        start: Timestamp of the first signal. Defaults to a fixed instant so
            two runs at the same seed are byte identical.
        interval_seconds: Gap between consecutive signals in event time.

    Returns:
        The generated signals in event time order.
    """
    origin = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    types = [
        SignalType.UNKNOWN,
        SignalType.EXPLOIT,
        SignalType.RECONNAISSANCE,
        SignalType.GENERIC,
    ]
    signals: list[Signal] = []

    for index in range(count):
        digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
        signals.append(
            Signal(
                signal_id=uuid4(),
                timestamp=origin + timedelta(seconds=interval_seconds * index),
                signal_type=types[digest[0] % len(types)],
                entity=EntityRef(
                    kind=EntityKind.DEVICE,
                    identifier=f"synthetic-device-{digest[1] % devices:02d}",
                ),
                anomaly_score=round(digest[2] / 255.0, 4),
                confidence=round(0.5 + (digest[3] / 255.0) * 0.5, 4),
                source="offline-replay",
                abstained=digest[4] % 10 == 0,
            )
        )
    return signals


@dataclass
class ReplayResult:
    """What one offline replay produced."""

    signals_ingested: int = 0
    situations_created: int = 0
    analyses_run: int = 0
    analyses_failed: int = 0
    iterations_logged: int = 0
    wall_clock_seconds: float = 0.0
    llm_calls: int = 0
    latency: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable view."""
        return {
            "signals_ingested": self.signals_ingested,
            "situations_created": self.situations_created,
            "analyses_run": self.analyses_run,
            "analyses_failed": self.analyses_failed,
            "iterations_logged": self.iterations_logged,
            "wall_clock_seconds": round(self.wall_clock_seconds, 4),
            "llm_calls": self.llm_calls,
            "latency": self.latency,
            "cache": self.cache,
        }


class OfflineReplay:
    """Drives signals through the reasoning stack without any transport.

    Args:
        llm_factory: Model factory. Use mock_llm_factory for fast deterministic
            runs and the Gemini factory, optionally cached, for real ones.
        run_log: Optional sink receiving one record per reasoning iteration.
        controls: Epistemic control set, which Level 9 varies.
        clock_mode: Time domain, which Level 8 varies.
        seed: Recorded into results and used by the synthetic generator.
        analyse_every: Run the graph once per this many ingested signals. The
            live server analyses on every signal, which for a long replay means
            thousands of near identical analyses; raising this samples the
            stream instead.
        correlation: Optional correlation strategy override.
        latency: Optional recorder. One is created when omitted.
    """

    def __init__(
        self,
        llm_factory: Callable[[], Any],
        *,
        run_log: RecordSink | None = None,
        controls: EpistemicControls | None = None,
        clock_mode: ClockMode | str = ClockMode.SEPARATED,
        seed: int = 42,
        analyse_every: int = 1,
        correlation: CorrelationStrategy | None = None,
        latency: LatencyRecorder | None = None,
        max_iterations: int | None = None,
        on_analysis: Callable[[Situation, dict[str, Any]], None] | None = None,
    ) -> None:
        self.llm_factory = llm_factory
        self.run_log = run_log
        self.controls = controls
        self.clock_mode = ClockMode(clock_mode)
        self.seed = seed
        self.analyse_every = max(1, analyse_every)
        self.max_iterations = max_iterations
        self.on_analysis = on_analysis
        self.latency = latency or LatencyRecorder()
        self.engine = ReasoningEngine()
        self.store = SituationStore(
            correlation=correlation,
            reasoning_engine=self.engine,
            clock_mode=self.clock_mode,
        )
        self.run_id = str(uuid4())

    def run(self, signals: Iterable[Signal]) -> ReplayResult:
        """Ingest every signal, analysing on the configured cadence."""
        result = ReplayResult()
        started = time.perf_counter()

        for index, signal in enumerate(signals):
            with self.latency.measure(Stage.INGEST):
                situation = self._ingest(signal)
            result.signals_ingested += 1

            if (index + 1) % self.analyse_every != 0:
                continue

            try:
                self._analyse(situation, result)
                result.analyses_run += 1
            except Exception as exc:
                result.analyses_failed += 1
                logger.error("Offline analysis failed: %s", exc, exc_info=True)

        result.wall_clock_seconds = time.perf_counter() - started
        result.situations_created = len(self.store._situations)
        result.latency = self.latency.summary()
        result.iterations_logged = getattr(self.run_log, "written", 0)
        result.llm_calls = self._model_call_count()
        return result

    def _model_call_count(self) -> int:
        """Report how many model calls were made, tolerating a failing factory.

        A factory that raises is a legitimate configuration: it is what an
        absent API key produces, and the generation node is built to fall back
        rather than fail. Reading a call count off it must not turn that
        supported path into a crash.
        """
        try:
            return getattr(self.llm_factory(), "calls", 0)
        except Exception:
            return 0

    def _ingest(self, signal: Signal) -> Situation:
        """Attach one signal, bypassing the async lock the live path needs.

        The offline replay is single threaded by construction, so taking the
        store's asyncio lock would add an event loop for no benefit. The
        correlation and attachment logic is the same code the server runs.
        """
        with self.latency.measure(Stage.ATTACH):
            situation = self.store._find_or_create(signal)
            situation.attach_evidence(signal)
        return situation

    def _analyse(self, situation: Situation, result: ReplayResult) -> None:
        """Run the full analysis pipeline for one situation."""
        with self.latency.measure(Stage.DETERMINISTIC):
            temporal = situation.temporal_snapshot()
            reasoning = self.engine.evaluate(situation)

        with self.latency.measure(Stage.LANGGRAPH):
            final_state = run_reasoning(
                situation,
                temporal,
                reasoning,
                llm_factory=self.llm_factory,
                controls=self.controls,
                run_log=self.run_log,
                run_id=self.run_id,
                max_iterations=self.max_iterations,
            )

        with self.latency.measure(Stage.EXPLANATION):
            build_explanation(final_state, reasoning, temporal)

        if self.on_analysis is not None:
            self.on_analysis(situation, final_state)

    def iter_situations(self) -> Iterator[Situation]:
        """Yield every situation the replay created."""
        return iter(self.store._situations.values())
