"""
Module: enigma_reason/observability/backpressure.py

Bounded concurrency and an observable backlog for the analysis fan out.

Section C4 of paper/EVIDENCE.md records that every accepted signal spawned an
unbounded asyncio task, each running a full reasoning pass costing seconds,
against a replay that emits roughly four records per second. The system was
oversubscribed by one to two orders of magnitude with no queue depth visible
anywhere, so the failure mode was silent: tasks accumulated in the event loop
and the only symptom was a dashboard that fell further behind.

This pool changes the failure mode from silent to measurable. Concurrency is
capped, the number of analyses waiting is counted, and work that arrives when
the backlog is already full is shed and counted rather than queued without
limit. Shedding is the honest behaviour: a stale analysis of a situation that
has since received twenty more signals has no value, and admitting it only
delays the analyses that do.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 4
DEFAULT_MAX_PENDING = 64


@dataclass(frozen=True)
class PoolSnapshot:
    """A point in time view of the analysis backlog."""

    max_concurrent: int
    max_pending: int
    running: int
    pending: int
    submitted: int
    completed: int
    shed: int
    failed: int
    peak_pending: int
    peak_running: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable view."""
        return asdict(self)


class AnalysisPool:
    """Runs analysis coroutines under a concurrency cap with a bounded backlog.

    Args:
        max_concurrent: How many analyses may run at once. The default of four
            reflects that each analysis occupies a thread pool slot for the
            duration of its language model calls.
        max_pending: How many analyses may wait for a slot before further
            submissions are shed.
        latency_recorder: Optional recorder that receives the time each
            analysis spends waiting for admission.
    """

    def __init__(
        self,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_pending: int = DEFAULT_MAX_PENDING,
        latency_recorder: Any | None = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.max_pending = max_pending
        self._latency = latency_recorder
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = 0
        self._pending = 0
        self.submitted = 0
        self.completed = 0
        self.shed = 0
        self.failed = 0
        self.peak_pending = 0
        self.peak_running = 0
        self._tasks: set[asyncio.Task] = set()

    @property
    def running(self) -> int:
        """Analyses currently executing."""
        return self._running

    @property
    def pending(self) -> int:
        """Analyses admitted but still waiting for a concurrency slot."""
        return self._pending

    def snapshot(self) -> PoolSnapshot:
        """Return the current backlog state."""
        return PoolSnapshot(
            max_concurrent=self.max_concurrent,
            max_pending=self.max_pending,
            running=self._running,
            pending=self._pending,
            submitted=self.submitted,
            completed=self.completed,
            shed=self.shed,
            failed=self.failed,
            peak_pending=self.peak_pending,
            peak_running=self.peak_running,
        )

    def submit(self, factory: Callable[[], Awaitable[None]]) -> bool:
        """Schedule an analysis, shedding it when the backlog is full.

        Args:
            factory: Zero argument callable producing the coroutine to run.
                A factory rather than a coroutine so that nothing is created
                when the submission is shed, which would otherwise leave an
                un awaited coroutine behind.

        Returns:
            True when the work was admitted, False when it was shed.
        """
        self.submitted += 1
        if self._pending >= self.max_pending:
            self.shed += 1
            logger.warning(
                "Analysis shed, backlog full at %d pending, %d shed in total",
                self._pending,
                self.shed,
            )
            return False

        self._pending += 1
        self.peak_pending = max(self.peak_pending, self._pending)
        task = asyncio.create_task(self._run(factory))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _run(self, factory: Callable[[], Awaitable[None]]) -> None:
        """Wait for a slot, then execute the analysis, always releasing state."""
        loop = asyncio.get_running_loop()
        queued_at = loop.time()
        try:
            async with self._semaphore:
                self._pending -= 1
                self._running += 1
                self.peak_running = max(self.peak_running, self._running)
                if self._latency is not None:
                    self._latency.record(
                        "analysis_admission_wait", (loop.time() - queued_at) * 1000.0
                    )
                try:
                    await factory()
                    self.completed += 1
                except Exception as exc:
                    self.failed += 1
                    logger.error("Analysis failed inside pool: %s", exc, exc_info=True)
                finally:
                    self._running -= 1
        except asyncio.CancelledError:
            self._pending = max(0, self._pending - 1)
            raise

    async def drain(self, timeout: float | None = None) -> bool:
        """Wait for every admitted analysis to finish.

        Args:
            timeout: Seconds to wait before giving up. None waits forever.

        Returns:
            True when the pool emptied, False when the timeout struck first.
        """
        if not self._tasks:
            return True
        pending = list(self._tasks)
        done, still_running = await asyncio.wait(pending, timeout=timeout)
        return not still_running
