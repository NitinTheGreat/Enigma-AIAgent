"""
Module: tests/test_level6_instrumentation.py

Tests for the Level 6 instrumentation.

The two constraints that matter most are covered here explicitly. Logging must
not change reasoning behaviour, which is asserted by running the same seeded
situation through a mocked model twice and comparing the final states field by
field. And the logger must never be the reason an analysis fails, which is
asserted by handing the runner a sink that raises on every call.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from enigma_reason.core.reasoning_engine import ReasoningEngine
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import EntityRef, Signal
from enigma_reason.domain.situation import Situation
from enigma_reason.foundation.clock import ClockMode
from enigma_reason.graph.runner import run_reasoning
from enigma_reason.observability.backpressure import AnalysisPool
from enigma_reason.observability.latency import LatencyRecorder, Stage, percentile
from enigma_reason.observability.llm_cache import (
    CachingLLMFactory,
    ResponseCache,
    prompt_key,
)
from enigma_reason.observability.manifest import build_run_manifest, environment_fingerprint
from enigma_reason.observability.run_log import (
    IterationRecorder,
    ListSink,
    RunLogWriter,
    text_hash,
)
from enigma_reason.replay.offline import (
    MockLLM,
    OfflineReplay,
    mock_llm_factory,
    synthetic_signals,
)


def build_situation(evidence: int = 6, seed: int = 42) -> Situation:
    """Return a situation carrying deterministic evidence."""
    situation = Situation(clock_mode=ClockMode.SEPARATED)
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(evidence):
        situation.attach_evidence(
            Signal(
                signal_id=uuid4(),
                timestamp=origin + timedelta(seconds=20 * index),
                signal_type=SignalType.EXPLOIT,
                entity=EntityRef(kind=EntityKind.DEVICE, identifier="device-01"),
                anomaly_score=0.6 + (index % 3) * 0.1,
                confidence=0.8,
                source="test-sensor",
                abstained=index % 5 == 0,
            )
        )
    return situation


def analyse(situation: Situation, run_log=None, seed: int = 42) -> dict:
    """Run the graph over a situation with a deterministic mocked model."""
    engine = ReasoningEngine()
    temporal = situation.temporal_snapshot()
    reasoning = engine.evaluate(situation)
    return run_reasoning(
        situation,
        temporal,
        reasoning,
        llm_factory=mock_llm_factory(seed=seed),
        run_log=run_log,
    )


class RaisingSink:
    """A sink that fails on every write, standing in for a broken log."""

    def __init__(self) -> None:
        self.attempts = 0

    def write(self, record: dict) -> None:
        self.attempts += 1
        raise OSError("sink deliberately unavailable")


PLACEHOLDER_ID = "normalised"


def comparable(state: dict) -> dict:
    """Strip the identifiers that differ between any two runs of one input.

    Situation and hypothesis identifiers come from uuid4, so a fresh fixture
    produces new ones on every call whether or not logging is enabled. Every
    value that reasoning actually computes is left intact. The control test
    below runs two unlogged analyses through this same normalisation to show it
    does not hide a genuine difference.
    """
    stripped = {k: v for k, v in state.items() if k != "context"}
    stripped["situation_id"] = PLACEHOLDER_ID
    stripped["hypotheses"] = [
        {**h, "hypothesis_id": f"position-{index}"}
        for index, h in enumerate(state.get("hypotheses", []))
    ]
    for snapshot in ("temporal_snapshot", "reasoning_snapshot"):
        if isinstance(stripped.get(snapshot), dict):
            stripped[snapshot] = {**stripped[snapshot], "situation_id": PLACEHOLDER_ID}
    return stripped


# ── Constraint: logging must not change reasoning behaviour ─────────────────


def test_two_unlogged_runs_agree_under_normalisation() -> None:
    """Control for the two tests below.

    If normalisation were hiding a real difference, this would still pass and
    prove nothing. What it establishes is the opposite direction: that two runs
    of the identical input agree on everything except the identifiers, so any
    disagreement the next tests find is attributable to logging.
    """
    first = analyse(build_situation(), run_log=None)
    second = analyse(build_situation(), run_log=None)
    assert comparable(first) == comparable(second)

    raw_ids_first = [h["hypothesis_id"] for h in first["hypotheses"]]
    raw_ids_second = [h["hypothesis_id"] for h in second["hypotheses"]]
    assert raw_ids_first != raw_ids_second


def test_logging_does_not_alter_final_state() -> None:
    """The same seed and mocked model must produce the same final state."""
    without = analyse(build_situation(), run_log=None)
    sink = ListSink()
    with_logging = analyse(build_situation(), run_log=sink)

    assert sink.written > 0
    assert comparable(without) == comparable(with_logging)


def test_logging_does_not_alter_hypothesis_confidences() -> None:
    """Confidence trajectories must match with logging on and off."""
    without = analyse(build_situation(evidence=9), run_log=None)
    with_logging = analyse(build_situation(evidence=9), run_log=ListSink())

    left = [(h["description"], h["confidence"]) for h in without["hypotheses"]]
    right = [(h["description"], h["confidence"]) for h in with_logging["hypotheses"]]
    assert left == right


def test_logging_does_not_alter_convergence_or_iteration_count() -> None:
    """The loop must terminate at the same point with logging on and off."""
    without = analyse(build_situation(evidence=9), run_log=None)
    with_logging = analyse(build_situation(evidence=9), run_log=ListSink())

    assert without["iteration_count"] == with_logging["iteration_count"]
    assert without["convergence_score"] == with_logging["convergence_score"]
    assert without["belief_stability_score"] == with_logging["belief_stability_score"]
    assert without["undecided_iterations"] == with_logging["undecided_iterations"]


# ── Constraint: the logger must not fail an analysis ────────────────────────


def test_failing_sink_does_not_break_analysis() -> None:
    """A sink that raises on every write must not propagate into reasoning."""
    sink = RaisingSink()
    state = analyse(build_situation(), run_log=sink)

    assert sink.attempts > 0
    assert state["iteration_count"] >= 1
    assert state["hypotheses"]


def test_unwritable_run_log_path_does_not_raise(tmp_path) -> None:
    """A writer pointed at an impossible path counts drops instead of raising."""
    writer = RunLogWriter(tmp_path / "missing" / "nested" / "run.jsonl")
    writer.write({"ok": True})
    writer.close()
    assert writer.written == 1


# ── Task 1: the structured run log ──────────────────────────────────────────


def test_run_log_emits_one_record_per_iteration() -> None:
    """Record count must equal the iteration count the graph reports."""
    sink = ListSink()
    state = analyse(build_situation(), run_log=sink)
    assert len(sink.records) == state["iteration_count"]


def test_run_log_record_carries_every_required_field() -> None:
    """Every field the brief names must be present on each record."""
    sink = ListSink()
    analyse(build_situation(), run_log=sink)

    required = {
        "run_id", "situation_id", "iteration", "wall_timestamp", "event_timestamp",
        "clock_mode", "trend", "convergence_score", "evidence_count",
        "source_diversity", "mean_anomaly", "burst_detected", "is_quiet",
        "abstained_evidence_count", "abstention_fraction", "hypotheses",
        "dominant_hypothesis_id", "dominant_iterations", "terminated",
        "termination_reason",
    }
    for record in sink.records:
        assert required <= set(record), required - set(record)

    hypothesis_fields = {
        "hypothesis_id", "text_hash", "confidence", "confidence_previous",
        "confidence_before_inertia", "inertia_clamped", "is_unknown", "status",
    }
    for record in sink.records:
        for hypothesis in record["hypotheses"]:
            assert hypothesis_fields <= set(hypothesis)


def test_only_the_last_record_is_terminated() -> None:
    """Termination is stamped once, on the record that closed the run."""
    sink = ListSink()
    analyse(build_situation(), run_log=sink)

    assert [r["terminated"] for r in sink.records[:-1]] == [False] * (len(sink.records) - 1)
    assert sink.records[-1]["terminated"] is True
    assert sink.records[-1]["termination_reason"] in {"converged", "max_iterations"}


def test_iterations_are_consecutive_from_one() -> None:
    """Iteration numbers must form a gapless sequence."""
    sink = ListSink()
    analyse(build_situation(), run_log=sink)
    assert [r["iteration"] for r in sink.records] == list(range(1, len(sink.records) + 1))


def test_event_and_wall_timestamps_are_distinct_domains() -> None:
    """Under replay the event clock must not follow wall time."""
    sink = ListSink()
    analyse(build_situation(), run_log=sink)
    record = sink.records[0]
    assert record["event_timestamp"].startswith("2026-01-01")
    assert not record["wall_timestamp"].startswith("2026-01-01T00:")
    assert record["clock_mode"] == ClockMode.SEPARATED.value


def test_text_hash_is_stable_and_normalising() -> None:
    """Whitespace and case must not register as a different hypothesis."""
    assert text_hash("Coordinated probing") == text_hash("  coordinated   PROBING ")
    assert text_hash("a") != text_hash("b")


def test_run_log_writer_produces_readable_jsonl(tmp_path) -> None:
    """Written records must round trip through JSON line by line."""
    path = tmp_path / "run.jsonl"
    with RunLogWriter(path) as writer:
        analyse(build_situation(), run_log=writer)
        written = writer.written

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == written
    for line in lines:
        assert "run_id" in json.loads(line)


def test_recorder_ignores_states_that_do_not_advance_iteration() -> None:
    """States repeated between super steps must not duplicate a record."""
    sink = ListSink()
    recorder = IterationRecorder(sink, situation_id="s", clock_mode="separated")
    recorder.observe({"iteration_count": 0, "hypotheses": []})
    recorder.observe({"iteration_count": 1, "hypotheses": []})
    recorder.observe({"iteration_count": 1, "hypotheses": []})
    recorder.observe({"iteration_count": 2, "hypotheses": []})
    recorder.finalise({"iteration_count": 2, "convergence_score": 0.0, "max_iterations": 2})
    assert [r["iteration"] for r in sink.records] == [1, 2]


# ── Task 2: per stage latency ───────────────────────────────────────────────


def test_percentiles_interpolate() -> None:
    """Percentiles must interpolate rather than pick a neighbouring sample."""
    samples = [float(v) for v in range(1, 101)]
    assert percentile(samples, 0.50) == pytest.approx(50.5)
    assert percentile(samples, 0.95) == pytest.approx(95.05)
    assert percentile(samples, 0.99) == pytest.approx(99.01)
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.99) == 7.0


def test_latency_summary_reports_stages_in_pipeline_order() -> None:
    """The report must read as the path a signal takes."""
    recorder = LatencyRecorder()
    for stage in (Stage.BROADCAST, Stage.INGEST, Stage.LANGGRAPH):
        recorder.record(stage, 1.0)
    assert list(recorder.summary()) == [
        Stage.INGEST.value, Stage.LANGGRAPH.value, Stage.BROADCAST.value,
    ]


def test_latency_measure_records_even_when_the_block_raises() -> None:
    """A failed stage must still contribute a timing."""
    recorder = LatencyRecorder()
    with pytest.raises(ValueError):
        with recorder.measure(Stage.LANGGRAPH):
            raise ValueError("stage failed")
    assert recorder.count(Stage.LANGGRAPH) == 1


# ── Task 3: backpressure ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pool_caps_concurrency() -> None:
    """Running analyses must never exceed the configured cap."""
    import asyncio

    pool = AnalysisPool(max_concurrent=3, max_pending=100)
    observed = []

    async def work() -> None:
        observed.append(pool.running)
        await asyncio.sleep(0.01)

    for _ in range(20):
        pool.submit(work)
    await pool.drain(timeout=10)

    assert max(observed) <= 3
    assert pool.completed == 20
    assert pool.shed == 0


@pytest.mark.asyncio
async def test_pool_sheds_when_backlog_is_full() -> None:
    """Submissions beyond the pending bound must be refused and counted."""
    import asyncio

    pool = AnalysisPool(max_concurrent=1, max_pending=4)

    async def work() -> None:
        await asyncio.sleep(0.05)

    outcomes = [pool.submit(work) for _ in range(20)]
    await pool.drain(timeout=10)

    assert outcomes.count(False) == pool.shed
    assert pool.shed > 0
    assert pool.submitted == 20
    assert pool.completed + pool.shed + pool.failed == 20


@pytest.mark.asyncio
async def test_pool_survives_a_failing_analysis() -> None:
    """One analysis raising must not stop the pool draining the rest."""
    import asyncio

    pool = AnalysisPool(max_concurrent=2, max_pending=50)

    async def boom() -> None:
        raise RuntimeError("analysis exploded")

    async def fine() -> None:
        await asyncio.sleep(0)

    for index in range(10):
        pool.submit(boom if index % 2 == 0 else fine)
    await pool.drain(timeout=10)

    assert pool.failed == 5
    assert pool.completed == 5


# ── Task 3: the evidence interval cache ─────────────────────────────────────


def test_event_intervals_are_memoised_until_evidence_changes() -> None:
    """Repeated reads within one analysis must not resort the evidence."""
    situation = build_situation(evidence=5)
    first = situation.event_intervals
    second = situation.event_intervals
    assert first == second
    assert first is not second

    before = len(situation.event_intervals)
    situation.attach_evidence(
        Signal(
            signal_id=uuid4(),
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            signal_type=SignalType.EXPLOIT,
            anomaly_score=0.5,
            confidence=0.5,
            source="test-sensor",
        )
    )
    assert len(situation.event_intervals) == before + 1


def test_event_intervals_still_sort_out_of_order_evidence() -> None:
    """The cache must not change the ordering guarantee."""
    situation = Situation(clock_mode=ClockMode.SEPARATED)
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for offset in (60, 0, 30):
        situation.attach_evidence(
            Signal(
                signal_id=uuid4(),
                timestamp=origin + timedelta(seconds=offset),
                signal_type=SignalType.GENERIC,
                anomaly_score=0.5,
                confidence=0.5,
                source="test-sensor",
            )
        )
    assert situation.event_intervals == [30.0, 30.0]


# ── Task 4: the run manifest ────────────────────────────────────────────────


def test_manifest_carries_all_three_repositories() -> None:
    """Traceability requires every repository, not just the agent."""
    started = datetime.now(timezone.utc)
    manifest = build_run_manifest(
        experiment="unit-test",
        seed=7,
        config={"clock_mode": "separated"},
        started_at=started,
    )
    assert set(manifest["git_commits"]) == {
        "Enigma-AIAgent", "Enigma-ML-Layer", "Enigma-Frontend",
    }
    assert manifest["seed"] == 7
    assert manifest["experiment"] == "unit-test"
    assert manifest["wall_clock_seconds"] >= 0


def test_manifest_hashes_a_named_dataset(tmp_path) -> None:
    """A dataset hash must reflect content, not path."""
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")

    started = datetime.now(timezone.utc)
    left = build_run_manifest(
        experiment="t", seed=1, config={}, started_at=started, dataset=first
    )
    right = build_run_manifest(
        experiment="t", seed=1, config={}, started_at=started, dataset=second
    )
    assert left["dataset"]["hash"] == right["dataset"]["hash"]
    assert left["dataset"]["hash"] != "absent"


def test_environment_fingerprint_names_the_interpreter() -> None:
    """The fingerprint must identify the interpreter and platform."""
    fingerprint = environment_fingerprint()
    assert fingerprint["python_version"].count(".") >= 1
    assert fingerprint["os"]
    assert "tensorflow_version" in fingerprint


# ── Task 5: offline replay ──────────────────────────────────────────────────


def test_offline_replay_runs_without_any_transport() -> None:
    """A replay must produce analyses with no server and no network."""
    sink = ListSink()
    replay = OfflineReplay(mock_llm_factory(seed=42), run_log=sink, seed=42)
    result = replay.run(synthetic_signals(60, seed=42, devices=6))

    assert result.signals_ingested == 60
    assert result.analyses_run == 60
    assert result.analyses_failed == 0
    assert result.situations_created > 0
    assert len(sink.records) >= result.analyses_run


def test_offline_replay_is_deterministic() -> None:
    """The same seed must produce the same situations and iteration counts."""
    def once() -> tuple[int, list[int]]:
        sink = ListSink()
        replay = OfflineReplay(mock_llm_factory(seed=42), run_log=sink, seed=42)
        result = replay.run(synthetic_signals(40, seed=42, devices=4))
        return result.situations_created, [r["iteration"] for r in sink.records]

    assert once() == once()


def test_mock_llm_returns_parseable_hypotheses() -> None:
    """The mock must satisfy the parser the real model output goes through."""
    model = MockLLM(seed=42)
    payload = json.loads(model.invoke("any prompt").content)
    assert len(payload) == 3
    assert any(item["is_benign"] for item in payload)
    for item in payload:
        assert 0.1 <= item["confidence"] <= 0.5


def test_mock_llm_is_a_pure_function_of_the_prompt() -> None:
    """Two configurations assembling the same prompt must see one answer."""
    model = MockLLM(seed=42)
    assert model.invoke("prompt a").content == model.invoke("prompt a").content
    assert model.invoke("prompt a").content != model.invoke("prompt b").content


# ── Task 6: response caching ────────────────────────────────────────────────


def test_cache_serves_a_repeated_prompt_without_calling_the_model() -> None:
    """A second identical prompt must not reach the model."""
    inner = MockLLM(seed=42)
    cache = ResponseCache(model="test-model")
    factory = CachingLLMFactory(lambda: inner, cache)

    first = factory().invoke("identical prompt")
    second = factory().invoke("identical prompt")

    assert first.content == second.content
    assert inner.calls == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


def test_cache_distinguishes_prompts_that_share_a_context() -> None:
    """Prior hypotheses change the prompt, so they must change the key."""
    base = "Situation metrics:\n- Evidence count: 5\n"
    assert prompt_key(base + "No prior hypotheses exist.") != prompt_key(
        base + "Prior active hypotheses (refine or replace):"
    )


def test_cache_persists_between_processes(tmp_path) -> None:
    """A saved cache must be reusable by a later run."""
    path = tmp_path / "cache.json"
    first = ResponseCache(path, model="m")
    first.put("prompt", "response")
    first.save()

    second = ResponseCache(path, model="m")
    assert second.get("prompt") == "response"
    assert second.stats.hits == 1


def test_cache_never_builds_the_model_when_every_prompt_hits(tmp_path) -> None:
    """A fully cached run must not require an API key."""
    path = tmp_path / "cache.json"
    seeded = ResponseCache(path, model="m")
    seeded.put("prompt", "response")
    seeded.save()

    def forbidden():
        raise AssertionError("model must not be constructed on a full cache")

    factory = CachingLLMFactory(forbidden, ResponseCache(path, model="m"))
    assert factory().invoke("prompt").content == "response"


def test_cached_factory_drives_a_real_replay(tmp_path) -> None:
    """The cache must work through the graph, not only in isolation."""
    path = tmp_path / "cache.json"
    inner = MockLLM(seed=42)
    cache = ResponseCache(path, model="mock")
    factory = CachingLLMFactory(lambda: inner, cache)

    OfflineReplay(factory, seed=42).run(synthetic_signals(20, seed=42, devices=2))
    calls_after_first = inner.calls
    cache.save()

    warm = ResponseCache(path, model="mock")
    warm_factory = CachingLLMFactory(lambda: inner, warm)
    OfflineReplay(warm_factory, seed=42).run(synthetic_signals(20, seed=42, devices=2))

    assert inner.calls == calls_after_first
    assert warm.stats.misses == 0
    assert warm.stats.hit_rate == 1.0
