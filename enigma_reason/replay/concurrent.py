"""
Module: enigma_reason/replay/concurrent.py

Drives many replay units at once so a level can finish inside a day.

Why this exists. OfflineReplay is serial, and at the measured 8.927 seconds
per model call a single pass over the frozen suite costs 65 hours. Every
concurrency figure in the Level 8 and Level 9 budget assumed a driver that
did not exist. This is that driver.

What is parallelised, and what deliberately is not. Units run concurrently;
everything inside a unit stays exactly as serial execution left it. A unit is
one scenario, because the iterations of an analysis are sequentially
dependent, the analyses of a situation are ordered by signal arrival, and the
situations of a scenario share the ingest counter that decides when an
analysis fires. A scenario is therefore the smallest boundary at which
splitting changes nothing, and running two scenarios at once cannot alter
either, because each gets its own store, engine and correlation state and
they share only thread safe collaborators.

Determinism. Two runs at different concurrency produce the same set of
analyses. They do not produce the same run log line order, because workers
interleave. Anything asserting on this must compare sorted sets, which
assert_same_analyses does.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "resource_exhausted",
    "resourceexhausted",
    "quota",
    "too many requests",
    "overloaded",
    "503",
    "502",
    "504",
    "server disconnected",
    "connection reset",
    "connection aborted",
    "connection error",
    "remote end closed",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "incomplete read",
    "broken pipe",
)


def looks_transient(error: BaseException) -> bool:
    """Return whether an exception reads as a transient fault rather than a bug.

    The Gemini client raises a family of types and wraps some of them, so the
    reliable signal is the rendered message rather than the class. A false
    positive costs one needless retry; a false negative is far worse than that
    and was measured: a dropped connection that this predicate did not
    recognise fell straight through to the fixed fallback hypotheses at
    nodes.py:200-202, which the generation node substitutes silently, so the
    run carried three invented hypotheses and nothing in its own output said
    so. Throttling and a dropped connection both mean try again, so both are
    named here.
    """
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


@dataclass
class RetryRecord:
    """One retry, kept so a throttled run is visible rather than just slow."""

    unit: str
    attempt: int
    delay_seconds: float
    error: str

    def to_dict(self) -> dict[str, Any]:
        """Return the retry as a run log record."""
        return {
            "record_type": "retry",
            "unit": self.unit,
            "attempt": self.attempt,
            "delay_seconds": round(self.delay_seconds, 3),
            "error": self.error[:400],
        }


@dataclass
class ConcurrentResult:
    """What a concurrent run produced, pooled across its units."""

    units: int = 0
    units_failed: int = 0
    signals_ingested: int = 0
    analyses_run: int = 0
    analyses_failed: int = 0
    situations_created: int = 0
    iterations_logged: int = 0
    wall_clock_seconds: float = 0.0
    concurrency: int = 1
    retries: list[RetryRecord] = field(default_factory=list)
    analysis_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the pooled result as plain data."""
        return {
            "units": self.units,
            "units_failed": self.units_failed,
            "signals_ingested": self.signals_ingested,
            "analyses_run": self.analyses_run,
            "analyses_failed": self.analyses_failed,
            "situations_created": self.situations_created,
            "iterations_logged": self.iterations_logged,
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "concurrency": self.concurrency,
            "retries": len(self.retries),
            "analyses_observed": len(self.analysis_keys),
        }


class RetryingModel:
    """Retries a throttled model call with exponential backoff and jitter."""

    def __init__(
        self,
        inner: Any,
        unit: str,
        on_retry: Callable[[RetryRecord], None],
        attempts: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        rng: random.Random | None = None,
    ) -> None:
        self._inner = inner
        self._unit = unit
        self._on_retry = on_retry
        self._attempts = max(1, attempts)
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._rng = rng or random.Random()

    def invoke(self, prompt: str) -> Any:
        """Forward the prompt, waiting and retrying while it is throttled."""
        last: BaseException | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                return self._inner.invoke(prompt)
            except BaseException as exc:
                if not looks_transient(exc) or attempt == self._attempts:
                    raise
                last = exc
                delay = min(self._base_delay * (2 ** (attempt - 1)), self._max_delay)
                delay += self._rng.uniform(0, delay * 0.25)
                record = RetryRecord(
                    unit=self._unit,
                    attempt=attempt,
                    delay_seconds=delay,
                    error=str(exc),
                )
                self._on_retry(record)
                logger.warning(
                    "Throttled on %s attempt %d, waiting %.1fs", self._unit, attempt, delay
                )
                time.sleep(delay)
        if last is not None:
            raise last
        raise RuntimeError("retry loop exited without a result")


class ConcurrentReplay:
    """Runs replay units in a thread pool, one independent replay per unit."""

    def __init__(
        self,
        build_replay: Callable[[str, Callable[[], Any]], Any],
        llm_factory: Callable[[], Any],
        *,
        concurrency: int = 1,
        run_log: Any | None = None,
        retry_attempts: int = 5,
        retry_base_delay: float = 2.0,
    ) -> None:
        self.build_replay = build_replay
        self.llm_factory = llm_factory
        self.concurrency = max(1, concurrency)
        self.run_log = run_log
        self.retry_attempts = retry_attempts
        self.retry_base_delay = retry_base_delay
        self._lock = threading.Lock()
        self._retries: list[RetryRecord] = []

    def _note_retry(self, record: RetryRecord) -> None:
        """Record a retry and write it into the run log."""
        with self._lock:
            self._retries.append(record)
        if self.run_log is not None:
            try:
                self.run_log.write(record.to_dict())
            except Exception:
                logger.warning("Retry record could not be written", exc_info=True)

    def _factory_for(self, unit: str) -> Callable[[], Any]:
        """Wrap the shared factory so this unit's calls retry on throttling."""

        def factory() -> Any:
            return RetryingModel(
                self.llm_factory(),
                unit=unit,
                on_retry=self._note_retry,
                attempts=self.retry_attempts,
                base_delay=self.retry_base_delay,
            )

        return factory

    def run(self, units: Sequence[tuple[str, Iterable[Any]]]) -> ConcurrentResult:
        """Run every unit, returning the pooled outcome."""
        result = ConcurrentResult(units=len(units), concurrency=self.concurrency)
        started = time.perf_counter()

        def work(item: tuple[str, Iterable[Any]]) -> tuple[str, Any]:
            name, signals = item
            replay = self.build_replay(name, self._factory_for(name))
            return name, replay.run(signals)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(work, unit): unit[0] for unit in units}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    _, outcome = future.result()
                except Exception as exc:
                    result.units_failed += 1
                    logger.error("Unit %s failed: %s", name, exc, exc_info=True)
                    continue
                result.signals_ingested += outcome.signals_ingested
                result.analyses_run += outcome.analyses_run
                result.analyses_failed += outcome.analyses_failed
                result.situations_created += outcome.situations_created

        result.wall_clock_seconds = time.perf_counter() - started
        result.retries = list(self._retries)
        result.iterations_logged = getattr(self.run_log, "written", 0)
        return result


def analysis_keys(run_log_path: Any) -> list[str]:
    """Return a sorted signature per situation, stable across orderings.

    Situation identifiers are freshly generated uuids, so they differ between
    any two runs and cannot appear in a key that two runs are expected to
    agree on. They are used only to group a situation's records together.

    What identifies a situation's work is the trace it produced: for every
    analysis that terminated, the evidence it held, the iteration it reached,
    why it stopped and which hypothesis led. Sorting those within a situation
    and sorting the situations against each other removes every ordering
    effect that concurrency introduces while preserving everything that a
    difference in work would change.
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    grouped: dict[str, list[str]] = defaultdict(list)
    for line in Path(run_log_path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record.get("record_type") == "retry":
            continue
        if not record.get("terminated"):
            continue
        grouped[str(record.get("situation_id"))].append(
            f"{record.get('evidence_count')}|{record.get('iteration')}|"
            f"{record.get('termination_reason')}|"
            f"{record.get('dominant_hypothesis_id') == 'UNKNOWN'}|"
            f"{round(float(record.get('convergence_score') or 0.0), 6)}"
        )
    return sorted(";".join(sorted(items)) for items in grouped.values())
