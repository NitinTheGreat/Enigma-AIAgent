"""Tests for Phase 4: Situation Reasoning Core.

Tests confidence computation, trend detection, reasoning engine output,
store-level reasoning summary, and determinism.
Uses clock patching via enigma_reason.domain.situation.utc_now.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest

from enigma_reason.core.reasoning_engine import (
    ConfidenceWeights,
    ReasoningEngine,
    TrendConfig,
)
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.reasoning import SituationReasoningSnapshot, Trend
from enigma_reason.domain.signal import Signal
from enigma_reason.domain.situation import Situation
from enigma_reason.store.situation_store import SituationStore

from tests.test_signal import _valid_signal


# ── Helpers ──────────────────────────────────────────────────────────────────

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _patched_now(dt: datetime):
    """Freeze utc_now() at the situation module level."""
    return patch("enigma_reason.domain.situation.utc_now", return_value=dt)


def _signal_at(ts: datetime, source: str = "src-a", score: float = 0.5, **kw) -> Signal:
    return Signal.model_validate(
        _valid_signal(timestamp=ts.isoformat(), source=source, anomaly_score=score, **kw)
    )


def _situation_with_signals(
    timestamps: list[datetime],
    sources: list[str] | None = None,
    scores: list[float] | None = None,
) -> Situation:
    """Create a situation with evidence at given timestamps/sources/scores."""
    sources = sources or ["src-a"] * len(timestamps)
    scores = scores or [0.5] * len(timestamps)
    with _patched_now(timestamps[0]):
        sit = Situation()
    for ts, src, sc in zip(timestamps, sources, scores):
        sig = _signal_at(ts, source=src, score=sc)
        with _patched_now(ts):
            sit.attach_evidence(sig)
    return sit


def _default_engine(**overrides) -> ReasoningEngine:
    return ReasoningEngine(**overrides)


# ── Confidence Tests ─────────────────────────────────────────────────────────


class TestConfidenceComputation:
    def test_empty_situation_zero_confidence(self) -> None:
        engine = _default_engine()
        with _patched_now(_BASE):
            sit = Situation()
            snap = engine.evaluate(sit)
        assert snap.confidence_level == 0.0

    def test_more_evidence_higher_confidence(self) -> None:
        """Monotonicity: adding evidence should not decrease confidence."""
        engine = _default_engine()
        confidences = []
        for n in range(1, 8):
            timestamps = [_BASE + timedelta(seconds=i * 10) for i in range(n)]
            sit = _situation_with_signals(timestamps)
            with _patched_now(timestamps[-1]):
                snap = engine.evaluate(sit)
            confidences.append(snap.confidence_level)
        # Each step should be >= previous (monotone non-decreasing)
        for i in range(1, len(confidences)):
            assert confidences[i] >= confidences[i - 1], (
                f"Confidence decreased at evidence count {i + 1}: "
                f"{confidences[i]} < {confidences[i - 1]}"
            )

    def test_higher_anomaly_scores_raise_confidence(self) -> None:
        engine = _default_engine()
        ts = [_BASE, _BASE + timedelta(seconds=30)]
        low = _situation_with_signals(ts, scores=[0.1, 0.1])
        high = _situation_with_signals(ts, scores=[0.9, 0.9])
        with _patched_now(ts[-1]):
            snap_low = engine.evaluate(low)
            snap_high = engine.evaluate(high)
        assert snap_high.confidence_level > snap_low.confidence_level

    def test_source_diversity_raises_confidence(self) -> None:
        engine = _default_engine()
        ts = [_BASE + timedelta(seconds=i * 10) for i in range(3)]
        single_src = _situation_with_signals(ts, sources=["a", "a", "a"])
        multi_src = _situation_with_signals(ts, sources=["a", "b", "c"])
        with _patched_now(ts[-1]):
            snap_single = engine.evaluate(single_src)
            snap_multi = engine.evaluate(multi_src)
        assert snap_multi.confidence_level > snap_single.confidence_level

    def test_confidence_clamped_to_unit_range(self) -> None:
        # Use extreme weights to try to exceed [0, 1]
        engine = ReasoningEngine(
            weights=ConfidenceWeights(
                evidence=10.0, rate=10.0, diversity=10.0, anomaly=10.0, burst=10.0,
            ),
        )
        ts = [_BASE + timedelta(seconds=i) for i in range(20)]
        sit = _situation_with_signals(ts, scores=[1.0] * 20)
        with _patched_now(ts[-1]):
            snap = engine.evaluate(sit)
        assert 0.0 <= snap.confidence_level <= 1.0

    def test_confidence_deterministic(self) -> None:
        """Same inputs → same output, every time."""
        engine = _default_engine()
        ts = [_BASE + timedelta(seconds=i * 5) for i in range(5)]
        sit = _situation_with_signals(ts)
        with _patched_now(ts[-1]):
            snap1 = engine.evaluate(sit)
            snap2 = engine.evaluate(sit)
        assert snap1.confidence_level == snap2.confidence_level


# ── Trend Detection Tests ────────────────────────────────────────────────────


class TestTrendDetection:
    def test_single_event_stable(self) -> None:
        engine = _default_engine()
        sit = _situation_with_signals([_BASE])
        with _patched_now(_BASE):
            snap = engine.evaluate(sit)
        assert snap.trend == Trend.STABLE

    def test_uniform_events_stable(self) -> None:
        engine = _default_engine()
        ts = [_BASE + timedelta(seconds=i * 10) for i in range(6)]
        sit = _situation_with_signals(ts)
        with _patched_now(ts[-1]):
            snap = engine.evaluate(sit)
        assert snap.trend == Trend.STABLE

    def test_burst_triggers_escalating(self) -> None:
        engine = _default_engine()
        # Slow start then rapid finish
        slow = [_BASE + timedelta(seconds=i * 60) for i in range(5)]
        fast_start = slow[-1] + timedelta(seconds=2)
        fast = [fast_start + timedelta(seconds=i * 2) for i in range(4)]
        sit = _situation_with_signals(slow + fast)
        with _patched_now(fast[-1]):
            snap = engine.evaluate(sit)
        assert snap.trend == Trend.ESCALATING
        assert snap.burst_detected is True

    def test_quiet_triggers_deescalating(self) -> None:
        engine = ReasoningEngine(quiet_window=timedelta(minutes=1))
        sit = _situation_with_signals([_BASE])
        # 5 minutes later — well past the 1-minute quiet window
        with _patched_now(_BASE + timedelta(minutes=5)):
            snap = engine.evaluate(sit)
        assert snap.trend == Trend.DEESCALATING
        assert snap.quiet_detected is True

    def test_deceleration_triggers_deescalating(self) -> None:
        """Events slowing down significantly → deescalating via interval comparison."""
        engine = ReasoningEngine(
            trend_config=TrendConfig(rate_fall_factor=1.5, recent_count=3),
            quiet_window=timedelta(hours=1),  # don't trigger quiet
        )
        # First 4 events: fast (5s apart)
        fast = [_BASE + timedelta(seconds=i * 5) for i in range(4)]
        # Last 3 events: very slow (120s apart)
        slow_start = fast[-1] + timedelta(seconds=120)
        slow = [slow_start + timedelta(seconds=i * 120) for i in range(3)]
        sit = _situation_with_signals(fast + slow)
        with _patched_now(slow[-1]):
            snap = engine.evaluate(sit)
        assert snap.trend == Trend.DEESCALATING

    def test_trend_transition_escalating_to_stable(self) -> None:
        """Burst period followed by uniform period → transitions from escalating to stable."""
        engine = _default_engine(quiet_window=timedelta(hours=1))
        # Phase 1: slow then burst → escalating
        slow = [_BASE + timedelta(seconds=i * 60) for i in range(5)]
        fast_start = slow[-1] + timedelta(seconds=2)
        fast = [fast_start + timedelta(seconds=i * 2) for i in range(4)]
        sit = _situation_with_signals(slow + fast)
        with _patched_now(fast[-1]):
            snap_esc = engine.evaluate(sit)
        assert snap_esc.trend == Trend.ESCALATING

        # Phase 2: extend with many uniform events at 30s intervals → stabilizes
        # 30s is well within the stable zone after the initial slow (60s) + fast (2s)
        uniform_start = fast[-1] + timedelta(seconds=30)
        uniform = [uniform_start + timedelta(seconds=i * 30) for i in range(40)]
        for ts in uniform:
            sig = _signal_at(ts)
            with _patched_now(ts):
                sit.attach_evidence(sig)
        with _patched_now(uniform[-1]):
            snap_stable = engine.evaluate(sit)
        assert snap_stable.trend == Trend.STABLE


# ── Reasoning Snapshot Tests ─────────────────────────────────────────────────


class TestReasoningSnapshot:
    def test_snapshot_is_immutable(self) -> None:
        engine = _default_engine()
        sit = _situation_with_signals([_BASE])
        with _patched_now(_BASE):
            snap = engine.evaluate(sit)
        assert isinstance(snap, SituationReasoningSnapshot)
        with pytest.raises(Exception):
            snap.evidence_count = 99

    def test_snapshot_fields_populated(self) -> None:
        engine = _default_engine()
        ts = [_BASE + timedelta(seconds=i * 10) for i in range(3)]
        sit = _situation_with_signals(ts, sources=["a", "b", "a"], scores=[0.3, 0.6, 0.9])
        with _patched_now(ts[-1]):
            snap = engine.evaluate(sit)
        assert snap.evidence_count == 3
        assert snap.source_diversity == 2
        assert snap.mean_anomaly_score == pytest.approx(0.6)
        assert snap.event_rate > 0.0
        assert snap.situation_id == str(sit.situation_id)

    def test_no_reasoning_for_empty_situation(self) -> None:
        engine = _default_engine()
        with _patched_now(_BASE):
            sit = Situation()
            snap = engine.evaluate(sit)
        assert snap.evidence_count == 0
        assert snap.confidence_level == 0.0
        assert snap.trend == Trend.STABLE


# ── Store Reasoning Summary Tests ────────────────────────────────────────────


class TestStoreReasoningSummary:
    @pytest.mark.asyncio
    async def test_empty_store_summary(self) -> None:
        engine = _default_engine()
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
            reasoning_engine=engine,
        )
        summary = await store.reasoning_summary()
        assert summary.total_situations == 0
        assert summary.escalating_situations == 0
        assert summary.stable_situations == 0
        assert summary.deescalating_situations == 0
        assert summary.average_confidence == 0.0

    @pytest.mark.asyncio
    async def test_summary_counts_trends(self) -> None:
        engine = _default_engine(quiet_window=timedelta(minutes=1))
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
            reasoning_engine=engine,
        )
        # Ingest one signal → situation will be quiet after 2 min → deescalating
        sig = _signal_at(_BASE)
        with _patched_now(_BASE):
            await store.ingest(sig)
        with _patched_now(_BASE + timedelta(minutes=2)):
            summary = await store.reasoning_summary()
        assert summary.total_situations == 1
        assert summary.deescalating_situations == 1

    @pytest.mark.asyncio
    async def test_summary_average_confidence(self) -> None:
        engine = _default_engine()
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
            reasoning_engine=engine,
        )
        # Ingest a signal
        sig = _signal_at(_BASE, score=0.8)
        with _patched_now(_BASE):
            await store.ingest(sig)
            summary = await store.reasoning_summary()
        assert summary.average_confidence > 0.0
        assert summary.max_confidence > 0.0

    @pytest.mark.asyncio
    async def test_summary_without_engine(self) -> None:
        """Store without reasoning_engine returns empty summary."""
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
        )
        summary = await store.reasoning_summary()
        assert summary.total_situations == 0

    @pytest.mark.asyncio
    async def test_summary_to_dict(self) -> None:
        engine = _default_engine()
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
            reasoning_engine=engine,
        )
        summary = await store.reasoning_summary()
        d = summary.to_dict()
        assert "escalating_situations" in d
        assert "stable_situations" in d
        assert "deescalating_situations" in d
        assert "average_confidence" in d
        assert "max_confidence" in d

    @pytest.mark.asyncio
    async def test_configurable_weights_affect_confidence(self) -> None:
        """Different weight configs produce different confidence values."""
        ts = [_BASE + timedelta(seconds=i * 10) for i in range(3)]

        engine_a = ReasoningEngine(weights=ConfidenceWeights(evidence=1.0, rate=0.0, diversity=0.0, anomaly=0.0, burst=0.0))
        engine_b = ReasoningEngine(weights=ConfidenceWeights(evidence=0.0, rate=0.0, diversity=0.0, anomaly=1.0, burst=0.0))

        store_a = SituationStore(ttl=timedelta(minutes=30), dormancy_window=timedelta(minutes=10), reasoning_engine=engine_a)
        store_b = SituationStore(ttl=timedelta(minutes=30), dormancy_window=timedelta(minutes=10), reasoning_engine=engine_b)

        for ts_val in ts:
            sig = _signal_at(ts_val, score=0.2)
            with _patched_now(ts_val):
                await store_a.ingest(sig)
                await store_b.ingest(sig)

        with _patched_now(ts[-1]):
            summary_a = await store_a.reasoning_summary()
            summary_b = await store_b.reasoning_summary()

        # Different weights → different confidences
        assert summary_a.average_confidence != summary_b.average_confidence
