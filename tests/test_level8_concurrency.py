"""
Tests for the concurrent replay driver and the cache safety it depends on.

The properties worth asserting here are the ones that only fail under load:
that two workers missing the same prompt produce one model call rather than
two, that a throttled call is retried and recorded rather than lost, and that
the analysis signature used to compare runs is blind to both worker ordering
and the freshly generated situation identifiers.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from enigma_reason.observability.llm_cache import CachingLLMFactory, ResponseCache
from enigma_reason.replay.concurrent import (
    ConcurrentReplay,
    RetryingModel,
    analysis_keys,
    looks_rate_limited,
)


class CountingModel:
    """Counts invocations and blocks long enough for a race to be real."""

    def __init__(self, delay: float = 0.05, reply: str = "answer") -> None:
        self.calls = 0
        self.delay = delay
        self.reply = reply
        self._lock = threading.Lock()

    def invoke(self, prompt: str):
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        return type("Reply", (), {"content": self.reply})()


class FlakyModel:
    """Fails with a throttling error a fixed number of times, then succeeds."""

    def __init__(self, failures: int, message: str = "429 rate limit exceeded") -> None:
        self.failures = failures
        self.calls = 0
        self.message = message

    def invoke(self, prompt: str):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(self.message)
        return type("Reply", (), {"content": "ok"})()


def test_concurrent_misses_on_one_prompt_produce_one_model_call():
    """Two workers racing on the same prompt must not both call the model."""
    model = CountingModel(delay=0.1)
    cache = ResponseCache(model="test")
    factory = CachingLLMFactory(lambda: model, cache)

    results: list[str] = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(factory().invoke("identical prompt").content)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert model.calls == 1
    assert results == ["answer"] * 8
    assert cache.stats.coalesced == 7
    assert cache.stats.hits == 7
    assert cache.stats.misses == 1


def test_distinct_prompts_still_each_reach_the_model():
    """Single flight must key on the prompt, not serialise everything."""
    model = CountingModel(delay=0.01)
    cache = ResponseCache(model="test")
    factory = CachingLLMFactory(lambda: model, cache)

    def worker(index: int):
        factory().invoke(f"prompt {index}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert model.calls == 6
    assert cache.stats.coalesced == 0


def test_cache_survives_concurrent_writes_without_losing_entries():
    """Every distinct prompt written under load must still be readable."""
    model = CountingModel(delay=0.0)
    cache = ResponseCache(model="test")
    factory = CachingLLMFactory(lambda: model, cache)

    def worker(index: int):
        for step in range(20):
            factory().invoke(f"p{index}-{step}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(cache) == 100
    for index in range(5):
        for step in range(20):
            assert cache.peek(f"p{index}-{step}") == "answer"


def test_persisted_cache_round_trips_after_concurrent_fill(tmp_path: Path):
    """A cache filled under load must be readable by the next process."""
    path = tmp_path / "cache.json"
    model = CountingModel(delay=0.0)
    cache = ResponseCache(path, model="test")
    factory = CachingLLMFactory(lambda: model, cache)

    def worker(index: int):
        factory().invoke(f"prompt {index}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    cache.save()

    reloaded = ResponseCache(path, model="test")
    assert len(reloaded) == 10


@pytest.mark.parametrize(
    "message",
    [
        "429 Too Many Requests",
        "RESOURCE_EXHAUSTED: quota exceeded",
        "The model is overloaded",
        "503 Service Unavailable",
    ],
)
def test_throttling_messages_are_recognised(message: str):
    """Every shape of throttling the client raises must be retried."""
    assert looks_rate_limited(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    ["invalid api key", "malformed request", "TypeError: bad argument"],
)
def test_non_throttling_errors_are_not_retried(message: str):
    """A genuine fault must surface rather than be retried into silence."""
    assert not looks_rate_limited(RuntimeError(message))


def test_retry_succeeds_after_throttling_and_records_each_attempt():
    """A throttled call retries, succeeds, and leaves a record per attempt."""
    recorded = []
    model = RetryingModel(
        FlakyModel(failures=2),
        unit="s00000",
        on_retry=recorded.append,
        attempts=5,
        base_delay=0.01,
    )
    assert model.invoke("prompt").content == "ok"
    assert len(recorded) == 2
    assert [r.attempt for r in recorded] == [1, 2]
    assert all(r.unit == "s00000" for r in recorded)
    assert all(r.delay_seconds > 0 for r in recorded)


def test_retry_gives_up_and_raises_after_the_attempt_budget():
    """Persistent throttling must fail loudly rather than return nothing."""
    recorded = []
    model = RetryingModel(
        FlakyModel(failures=99),
        unit="s00001",
        on_retry=recorded.append,
        attempts=3,
        base_delay=0.01,
    )
    with pytest.raises(RuntimeError):
        model.invoke("prompt")
    assert len(recorded) == 2


def test_retry_does_not_swallow_a_genuine_fault():
    """A non throttling error must reach the caller on the first attempt."""
    recorded = []
    model = RetryingModel(
        FlakyModel(failures=1, message="invalid api key"),
        unit="s00002",
        on_retry=recorded.append,
        attempts=5,
        base_delay=0.01,
    )
    with pytest.raises(RuntimeError):
        model.invoke("prompt")
    assert recorded == []


def test_retry_backoff_grows_between_attempts():
    """Successive waits must increase so a throttled run backs off."""
    recorded = []
    model = RetryingModel(
        FlakyModel(failures=3),
        unit="s00003",
        on_retry=recorded.append,
        attempts=6,
        base_delay=0.01,
    )
    model.invoke("prompt")
    delays = [r.delay_seconds for r in recorded]
    assert delays[0] < delays[-1]


def _write_log(path: Path, situation: str, rows: list[tuple[int, int]]) -> None:
    """Write a minimal terminated run log for signature tests."""
    with path.open("w", encoding="utf-8") as handle:
        for evidence, iteration in rows:
            handle.write(
                json.dumps(
                    {
                        "situation_id": situation,
                        "evidence_count": evidence,
                        "iteration": iteration,
                        "terminated": True,
                        "termination_reason": "max_iterations",
                        "dominant_hypothesis_id": "abc",
                        "convergence_score": 0.25,
                    }
                )
                + "\n"
            )


def test_analysis_signature_ignores_situation_identifiers(tmp_path: Path):
    """Two runs differing only in generated uuids must compare equal."""
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_log(first, "uuid-one", [(1, 3), (2, 3)])
    _write_log(second, "uuid-two", [(1, 3), (2, 3)])
    assert analysis_keys(first) == analysis_keys(second)


def test_analysis_signature_ignores_record_order(tmp_path: Path):
    """Worker interleaving must not change the signature."""
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_log(first, "s", [(1, 3), (2, 3), (3, 3)])
    _write_log(second, "s", [(3, 3), (1, 3), (2, 3)])
    assert analysis_keys(first) == analysis_keys(second)


def test_analysis_signature_notices_different_work(tmp_path: Path):
    """A genuinely different set of analyses must not compare equal."""
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_log(first, "s", [(1, 3), (2, 3)])
    _write_log(second, "s", [(1, 3), (9, 3)])
    assert analysis_keys(first) != analysis_keys(second)


def test_analysis_signature_skips_retry_records(tmp_path: Path):
    """Retry records are not analyses and must not enter the signature."""
    path = tmp_path / "a.jsonl"
    _write_log(path, "s", [(1, 3)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"record_type": "retry", "unit": "s00000", "attempt": 1,
                 "delay_seconds": 2.0, "error": "429"}
            )
            + "\n"
        )
    assert len(analysis_keys(path)) == 1


class _StubReplay:
    """Minimal replay recording which unit it ran."""

    def __init__(self, seen: list[str], unit: str) -> None:
        self._seen = seen
        self._unit = unit

    def run(self, signals):
        self._seen.append(self._unit)
        return type(
            "Result",
            (),
            {
                "signals_ingested": len(list(signals)),
                "analyses_run": 1,
                "analyses_failed": 0,
                "situations_created": 1,
            },
        )()


def test_concurrent_driver_runs_every_unit():
    """No unit may be dropped by the pool."""
    seen: list[str] = []
    driver = ConcurrentReplay(
        lambda unit, factory: _StubReplay(seen, unit),
        lambda: CountingModel(),
        concurrency=4,
    )
    units = [(f"s{i:05d}", [1, 2, 3]) for i in range(12)]
    result = driver.run(units)
    assert sorted(seen) == sorted(u[0] for u in units)
    assert result.units == 12
    assert result.units_failed == 0
    assert result.analyses_run == 12


def test_concurrent_driver_survives_one_failing_unit():
    """One bad unit must not take the whole run down."""

    def build(unit: str, factory):
        if unit == "s00003":
            raise RuntimeError("unit exploded")
        return _StubReplay([], unit)

    driver = ConcurrentReplay(build, lambda: CountingModel(), concurrency=4)
    result = driver.run([(f"s{i:05d}", [1]) for i in range(6)])
    assert result.units_failed == 1
    assert result.analyses_run == 5
