"""Tests for Phase 2: Temporal Awareness.

Tests temporal metrics, burst/quiet detection, and temporal snapshots.
Uses clock patching via enigma_reason.foundation.clock.utc_now.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest

from enigma_reason.domain.enums import SignalType
from enigma_reason.domain.signal import Signal
from enigma_reason.domain.situation import Situation
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.store.situation_store import SituationStore

from tests.test_signal import _valid_signal


# ── Helpers ──────────────────────────────────────────────────────────────────

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _signal_at(ts: datetime, **overrides) -> Signal:
    """Create a validated Signal with a specific timestamp."""
    return Signal.model_validate(_valid_signal(timestamp=ts.isoformat(), **overrides))


def _patched_now(dt: datetime):
    """Context manager to freeze utc_now() everywhere it's imported."""
    return patch("enigma_reason.domain.situation.utc_now", return_value=dt)


def _situation_with_events(timestamps: list[datetime]) -> Situation:
    """Create a situation and attach evidence at each timestamp."""
    with _patched_now(timestamps[0]):
        sit = Situation()
    for ts in timestamps:
        sig = _signal_at(ts)
        with _patched_now(ts):
            sit.attach_evidence(sig)
    return sit


# ── Temporal Metric Tests ────────────────────────────────────────────────────


class TestTemporalMetrics:
    def test_first_seen_at_with_no_evidence(self) -> None:
        sit = Situation()
        assert sit.first_seen_at is None

    def test_first_and_last_seen_at(self) -> None:
        t1 = _BASE
        t2 = _BASE + timedelta(minutes=5)
        t3 = _BASE + timedelta(minutes=10)
        sit = _situation_with_events([t1, t2, t3])
        assert sit.first_seen_at == t1
        assert sit.last_seen_at == t3

    def test_first_last_seen_single_event(self) -> None:
        sit = _situation_with_events([_BASE])
        assert sit.first_seen_at == _BASE
        assert sit.last_seen_at == _BASE

    def test_active_duration_no_events(self) -> None:
        sit = Situation()
        assert sit.active_duration == 0.0

    def test_active_duration_single_event(self) -> None:
        sit = _situation_with_events([_BASE])
        assert sit.active_duration == 0.0

    def test_active_duration_multiple_events(self) -> None:
        t1, t2 = _BASE, _BASE + timedelta(minutes=10)
        sit = _situation_with_events([t1, t2])
        assert sit.active_duration == 600.0  # 10 minutes

    def test_event_intervals_empty(self) -> None:
        sit = Situation()
        assert sit.event_intervals == []

    def test_event_intervals_single_event(self) -> None:
        sit = _situation_with_events([_BASE])
        assert sit.event_intervals == []

    def test_event_intervals_correct_order(self) -> None:
        t1 = _BASE
        t2 = _BASE + timedelta(seconds=30)
        t3 = _BASE + timedelta(seconds=90)
        sit = _situation_with_events([t1, t2, t3])
        intervals = sit.event_intervals
        assert len(intervals) == 2
        assert intervals[0] == pytest.approx(30.0)
        assert intervals[1] == pytest.approx(60.0)

    def test_event_intervals_sorted_regardless_of_attach_order(self) -> None:
        """Even if signals are attached out of order, intervals use sorted timestamps."""
        t1 = _BASE
        t2 = _BASE + timedelta(seconds=20)
        t3 = _BASE + timedelta(seconds=50)
        # Attach out of order: t3, t1, t2
        with _patched_now(t1):
            sit = Situation()
        for ts in [t3, t1, t2]:
            with _patched_now(ts):
                sit.attach_evidence(_signal_at(ts))
        intervals = sit.event_intervals
        assert intervals[0] == pytest.approx(20.0)
        assert intervals[1] == pytest.approx(30.0)

    def test_event_rate_no_duration(self) -> None:
        sit = _situation_with_events([_BASE])
        assert sit.event_rate == 0.0

    def test_event_rate_calculation(self) -> None:
        # 6 events over 3 minutes → 2 events/minute
        timestamps = [_BASE + timedelta(seconds=i * 30) for i in range(7)]
        sit = _situation_with_events(timestamps)
        # active_duration = 180s, 7 events → 7/180*60 = ~2.33 events/min
        assert sit.event_rate == pytest.approx(7.0 / 180.0 * 60.0, rel=1e-6)

    def test_event_rate_two_events(self) -> None:
        t1, t2 = _BASE, _BASE + timedelta(minutes=1)
        sit = _situation_with_events([t1, t2])
        assert sit.event_rate == pytest.approx(2.0)  # 2 events / 1 minute


# ── Burst Detection Tests ───────────────────────────────────────────────────


class TestBurstDetection:
    def test_no_burst_with_insufficient_data(self) -> None:
        """Less than recent_count+1 events → never bursting."""
        sit = _situation_with_events([_BASE, _BASE + timedelta(seconds=10)])
        assert not sit.is_bursting(burst_factor=3.0, recent_count=3)

    def test_no_burst_uniform_intervals(self) -> None:
        """Uniform spacing → no burst."""
        timestamps = [_BASE + timedelta(seconds=i * 10) for i in range(8)]
        sit = _situation_with_events(timestamps)
        assert not sit.is_bursting(burst_factor=3.0, recent_count=3)

    def test_burst_detected_on_acceleration(self) -> None:
        """Slow start then rapid finish should trigger burst."""
        # First 5 events: 60s apart (slow)
        slow = [_BASE + timedelta(seconds=i * 60) for i in range(5)]
        # Last 4 events: 2s apart (rapid) — burst_factor=3 should fire
        fast_start = slow[-1] + timedelta(seconds=2)
        fast = [fast_start + timedelta(seconds=i * 2) for i in range(4)]
        sit = _situation_with_events(slow + fast)
        assert sit.is_bursting(burst_factor=3.0, recent_count=3)

    def test_burst_factor_configurable(self) -> None:
        """Higher burst_factor makes burst harder to trigger."""
        slow = [_BASE + timedelta(seconds=i * 30) for i in range(5)]
        fast_start = slow[-1] + timedelta(seconds=10)
        fast = [fast_start + timedelta(seconds=i * 10) for i in range(4)]
        sit = _situation_with_events(slow + fast)
        # With factor=1.0 (any acceleration = burst) → might fire
        # With factor=100.0 → effectively impossible
        assert not sit.is_bursting(burst_factor=100.0, recent_count=3)

    def test_burst_returns_false_with_zero_overall_mean(self) -> None:
        """All events at the exact same timestamp → no burst."""
        sit = _situation_with_events([_BASE] * 5)
        assert not sit.is_bursting(burst_factor=3.0, recent_count=3)


# ── Quiet Detection Tests ───────────────────────────────────────────────────


class TestQuietDetection:
    def test_quiet_with_no_evidence(self) -> None:
        sit = Situation()
        assert sit.is_quiet(quiet_window=timedelta(minutes=5))

    def test_not_quiet_when_recent_evidence(self) -> None:
        t1 = _BASE
        with _patched_now(t1):
            sit = _situation_with_events([t1])
        # "Now" is the same as last event → not quiet
        with _patched_now(t1 + timedelta(seconds=30)):
            assert not sit.is_quiet(quiet_window=timedelta(minutes=5))

    def test_quiet_after_window_elapses(self) -> None:
        t1 = _BASE
        with _patched_now(t1):
            sit = _situation_with_events([t1])
        # 10 minutes later with 5-minute quiet window → quiet
        with _patched_now(t1 + timedelta(minutes=10)):
            assert sit.is_quiet(quiet_window=timedelta(minutes=5))

    def test_quiet_window_configurable(self) -> None:
        t1 = _BASE
        with _patched_now(t1):
            sit = _situation_with_events([t1])
        future = t1 + timedelta(minutes=3)
        with _patched_now(future):
            assert sit.is_quiet(quiet_window=timedelta(minutes=2))
            assert not sit.is_quiet(quiet_window=timedelta(minutes=5))


# ── Temporal Snapshot Tests ──────────────────────────────────────────────────


class TestTemporalSnapshot:
    def test_snapshot_is_frozen(self) -> None:
        sit = _situation_with_events([_BASE])
        with _patched_now(_BASE):
            snap = sit.temporal_snapshot()
        assert isinstance(snap, SituationTemporalSnapshot)
        with pytest.raises(Exception):
            snap.event_count = 99  # frozen model

    def test_snapshot_fields(self) -> None:
        t1 = _BASE
        t2 = _BASE + timedelta(seconds=60)
        sit = _situation_with_events([t1, t2])
        with _patched_now(t2):
            snap = sit.temporal_snapshot(
                burst_factor=3.0, recent_count=3, quiet_window=timedelta(minutes=5)
            )
        assert snap.event_count == 2
        assert snap.active_duration_seconds == pytest.approx(60.0)
        assert snap.event_rate_per_minute == pytest.approx(2.0)
        assert snap.last_event_age_seconds == pytest.approx(0.0)
        assert snap.mean_interval_seconds == pytest.approx(60.0)
        assert snap.burst_detected is False
        assert snap.quiet_detected is False

    def test_snapshot_captures_burst(self) -> None:
        slow = [_BASE + timedelta(seconds=i * 60) for i in range(5)]
        fast_start = slow[-1] + timedelta(seconds=2)
        fast = [fast_start + timedelta(seconds=i * 2) for i in range(4)]
        sit = _situation_with_events(slow + fast)
        with _patched_now(fast[-1]):
            snap = sit.temporal_snapshot(burst_factor=3.0, recent_count=3)
        assert snap.burst_detected is True

    def test_snapshot_captures_quiet(self) -> None:
        sit = _situation_with_events([_BASE])
        with _patched_now(_BASE + timedelta(minutes=10)):
            snap = sit.temporal_snapshot(quiet_window=timedelta(minutes=5))
        assert snap.quiet_detected is True


# ── Store Temporal Summary Tests ─────────────────────────────────────────────


class TestTemporalSummary:
    @pytest.mark.asyncio
    async def test_empty_store_summary(self) -> None:
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
        )
        summary = await store.temporal_summary()
        assert summary.total_situations == 0
        assert summary.bursting_situations == 0
        assert summary.quiet_situations == 0
        assert summary.max_event_rate == 0.0

    @pytest.mark.asyncio
    async def test_summary_counts_quiet(self) -> None:
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
            quiet_window=timedelta(minutes=1),
        )
        sig = _signal_at(_BASE)
        with _patched_now(_BASE):
            await store.ingest(sig)
        # 2 minutes later — situation is quiet
        with _patched_now(_BASE + timedelta(minutes=2)):
            summary = await store.temporal_summary()
        assert summary.total_situations == 1
        assert summary.quiet_situations == 1

    @pytest.mark.asyncio
    async def test_summary_tracks_max_event_rate(self) -> None:
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
        )
        # Ingest 3 signals rapidly for one situation
        for i in range(3):
            ts = _BASE + timedelta(seconds=i * 10)
            sig = _signal_at(ts)
            with _patched_now(ts):
                await store.ingest(sig)
        with _patched_now(_BASE + timedelta(seconds=20)):
            summary = await store.temporal_summary()
        assert summary.max_event_rate > 0.0

    @pytest.mark.asyncio
    async def test_summary_to_dict(self) -> None:
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
        )
        summary = await store.temporal_summary()
        d = summary.to_dict()
        assert "total_situations" in d
        assert "bursting_situations" in d
        assert "quiet_situations" in d
        assert "max_event_rate" in d
