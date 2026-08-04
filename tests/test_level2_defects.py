"""
Module: tests/test_level2_defects.py

Regression tests for the six defects remediated in Level 2, plus the ablation
switches Level 9 depends on.

Each defect has at least one test that fails against the old behaviour and
passes against the new. Where the old behaviour is still reachable, by way of
clock_mode or a disabled switch, the test asserts the old behaviour on that path
too, so the parameterisation itself is verified rather than assumed.

Defect references are to paper/EVIDENCE.md sections D1 to D6.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from enigma_reason.adapters.unsw_threat import UNSWThreatAdapter
from enigma_reason.core.reasoning_engine import ReasoningEngine
from enigma_reason.domain.hypothesis import UNKNOWN_HYPOTHESIS_ID
from enigma_reason.domain.reasoning import SituationReasoningSnapshot, Trend
from enigma_reason.domain.situation import Situation
from enigma_reason.foundation.clock import (
    ClockMode,
    ReplayClock,
    WallClock,
    make_clock,
)
from enigma_reason.graph.builder import EpistemicControls, build_reasoning_graph
from enigma_reason.graph.nodes import (
    hypothesis_sanity_gate,
    make_apply_belief_inertia,
    make_evaluate_hypotheses,
    make_generate_hypotheses,
    make_update_convergence,
)
from enigma_reason.store.situation_store import SituationStore

from tests.test_signal import _valid_signal
from enigma_reason.domain.signal import Signal


_BASE = datetime(2015, 2, 18, 12, 0, 0, tzinfo=timezone.utc)


def _signal_at(timestamp: datetime, **overrides) -> Signal:
    """Build a validated Signal carrying a specific event timestamp."""
    return Signal.model_validate(
        _valid_signal(timestamp=timestamp.isoformat(), **overrides)
    )


def _hypothesis(
    description: str,
    confidence: float,
    hypothesis_id: str | None = None,
    status: str = "active",
    **extra,
) -> dict:
    """Build a hypothesis dict matching the graph's working representation."""
    base = {
        "hypothesis_id": hypothesis_id or description[:12],
        "description": description,
        "confidence": confidence,
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "status": status,
        "belief_velocity": 0.0,
        "belief_acceleration": 0.0,
        "dominant_iterations": 0,
    }
    base.update(extra)
    return base


def _state(hypotheses: list[dict], **overrides) -> dict:
    """Build a minimal reasoning state for node-level tests."""
    state = {
        "situation_id": "level2",
        "temporal_snapshot": {},
        "reasoning_snapshot": SituationReasoningSnapshot(
            situation_id="level2",
            evidence_count=6,
            event_rate=2.0,
            burst_detected=False,
            quiet_detected=False,
            confidence_level=0.5,
            trend=Trend.STABLE,
            source_diversity=2,
            mean_anomaly_score=0.5,
        ).model_dump(),
        "context": {
            "evidence_count": 6,
            "event_rate_per_minute": 2.0,
            "active_duration_seconds": 180.0,
            "burst_detected": False,
            "quiet_detected": False,
            "trend": "stable",
            "confidence_level": 0.5,
            "source_diversity": 2,
            "mean_anomaly_score": 0.5,
            "iteration": 0,
        },
        "hypotheses": hypotheses,
        "iteration_count": 0,
        "convergence_score": 0.0,
        "max_iterations": 3,
        "convergence_threshold": 0.8,
        "belief_stability_score": 0.0,
        "undecided_iterations": 0,
        "last_confidence_shift": 0.0,
        "convergence_persistence": 2,
        "max_confidence_delta": 0.15,
    }
    state.update(overrides)
    return state


# ── D1: mixed clock domains ─────────────────────────────────────────────────


class TestClockDomains:
    def test_make_clock_returns_expected_implementations(self) -> None:
        assert isinstance(make_clock(ClockMode.SEPARATED), ReplayClock)
        assert isinstance(make_clock(ClockMode.CONFLATED), WallClock)
        assert isinstance(make_clock(ClockMode.WALL), WallClock)

    def test_replay_clock_advances_monotonically(self) -> None:
        clock = ReplayClock()
        clock.observe(_BASE)
        clock.observe(_BASE - timedelta(days=1))
        assert clock.latest_observed == _BASE
        clock.observe(_BASE + timedelta(hours=1))
        assert clock.latest_observed == _BASE + timedelta(hours=1)

    def test_separated_is_not_quiet_from_host_time_alone(self) -> None:
        """The defect: eleven years of host time must not imply quiet."""
        clock = ReplayClock()
        situation = Situation(clock=clock, clock_mode=ClockMode.SEPARATED)
        situation.attach_evidence(_signal_at(_BASE))
        assert situation.is_quiet(timedelta(minutes=5)) is False

    def test_conflated_reports_quiet_from_host_time_alone(self) -> None:
        """The defect stays reachable so Level 8 can measure it."""
        situation = Situation(clock_mode=ClockMode.CONFLATED)
        situation.attach_evidence(_signal_at(_BASE))
        assert situation.is_quiet(timedelta(minutes=5)) is True

    def test_separated_reports_quiet_when_event_time_advances(self) -> None:
        clock = ReplayClock()
        situation = Situation(clock=clock, clock_mode=ClockMode.SEPARATED)
        situation.attach_evidence(_signal_at(_BASE))
        clock.observe(_BASE + timedelta(minutes=30))
        assert situation.is_quiet(timedelta(minutes=5)) is True

    def test_trend_is_not_pinned_to_deescalating_under_separated(self) -> None:
        """The downstream consequence of D1 was a constant trend label."""
        engine = ReasoningEngine(
            quiet_window=timedelta(minutes=5), clock_mode=ClockMode.SEPARATED
        )
        clock = ReplayClock()
        situation = Situation(clock=clock, clock_mode=ClockMode.SEPARATED)
        for offset in range(6):
            situation.attach_evidence(
                _signal_at(_BASE + timedelta(seconds=offset * 10))
            )
        snapshot = engine.evaluate(situation)
        assert snapshot.quiet_detected is False
        assert snapshot.trend is not Trend.DEESCALATING

    def test_trend_is_pinned_to_deescalating_under_conflated(self) -> None:
        engine = ReasoningEngine(
            quiet_window=timedelta(minutes=5), clock_mode=ClockMode.CONFLATED
        )
        situation = Situation(clock_mode=ClockMode.CONFLATED)
        for offset in range(6):
            situation.attach_evidence(
                _signal_at(_BASE + timedelta(seconds=offset * 10))
            )
        snapshot = engine.evaluate(situation)
        assert snapshot.quiet_detected is True
        assert snapshot.trend is Trend.DEESCALATING

    @pytest.mark.asyncio
    async def test_store_shares_one_clock_across_situations(self) -> None:
        """A per-situation replay clock could never report anything quiet."""
        store = SituationStore(clock_mode=ClockMode.SEPARATED)
        first = await store.ingest(_signal_at(_BASE, entity={"kind": "device", "identifier": "a"}))
        second = await store.ingest(
            _signal_at(_BASE + timedelta(minutes=30), entity={"kind": "device", "identifier": "b"})
        )
        assert first.clock is second.clock
        assert first.is_quiet(timedelta(minutes=5)) is True
        assert second.is_quiet(timedelta(minutes=5)) is False

    def test_engine_rejects_mismatched_situation(self) -> None:
        engine = ReasoningEngine(clock_mode=ClockMode.SEPARATED)
        situation = Situation(clock_mode=ClockMode.WALL)
        situation.attach_evidence(_signal_at(_BASE))
        with pytest.raises(ValueError, match="clock mode mismatch"):
            engine.evaluate(situation)


# ── D2: belief inertia ──────────────────────────────────────────────────────


class TestBeliefInertiaClamps:
    def test_raw_delta_of_025_is_clamped_to_010(self) -> None:
        """DONE CHECK item 2."""
        hypothesis = _hypothesis(
            "Rapid belief escalation hypothesis",
            0.55,
            confidence_previous=0.30,
            confidence_before_inertia=0.55,
        )
        node = make_apply_belief_inertia(max_confidence_delta=0.10)
        result = node(_state([hypothesis]))
        updated = result["hypotheses"][0]
        assert updated["confidence_before_inertia"] == pytest.approx(0.55)
        assert updated["confidence"] == pytest.approx(0.40)
        assert updated["inertia_clamped"] is True

    def test_inertia_writes_confidence_not_only_velocity(self) -> None:
        """The old node wrote velocity alone, so confidence never moved."""
        hypothesis = _hypothesis(
            "Belief inertia writes confidence",
            0.80,
            confidence_previous=0.20,
            confidence_before_inertia=0.80,
        )
        node = make_apply_belief_inertia(max_confidence_delta=0.15)
        result = node(_state([hypothesis]))
        assert result["hypotheses"][0]["confidence"] != pytest.approx(0.80)

    def test_disabled_cap_leaves_confidence_untouched(self) -> None:
        hypothesis = _hypothesis(
            "Unbounded belief hypothesis",
            0.95,
            confidence_previous=0.05,
            confidence_before_inertia=0.95,
        )
        node = make_apply_belief_inertia(max_confidence_delta=float("inf"))
        result = node(_state([hypothesis]))
        assert result["hypotheses"][0]["confidence"] == pytest.approx(0.95)
        assert result["hypotheses"][0]["inertia_clamped"] is False


# ── D3: sanity gate penalty ─────────────────────────────────────────────────


class TestSanityGatePenalty:
    def test_vague_hypothesis_penalty_is_applied(self) -> None:
        """DONE CHECK item 3. The old branch mutated a discarded copy."""
        hypotheses = [
            _hypothesis("Maybe something anomalous is occurring here", 0.40),
            _hypothesis("Normal operational traffic within parameters", 0.40),
        ]
        before = copy.deepcopy(hypotheses)
        result = hypothesis_sanity_gate(_state(hypotheses))
        vague = next(
            h for h in result["hypotheses"] if h["description"].startswith("Maybe")
        )
        assert before[0]["confidence"] == pytest.approx(0.40)
        assert vague["confidence"] == pytest.approx(0.30)

    def test_non_vague_hypothesis_is_untouched(self) -> None:
        hypotheses = [
            _hypothesis("Coordinated scanning from a single origin", 0.40),
            _hypothesis("Normal operational traffic within parameters", 0.40),
        ]
        result = hypothesis_sanity_gate(_state(hypotheses))
        precise = next(
            h for h in result["hypotheses"] if h["description"].startswith("Coordinated")
        )
        assert precise["confidence"] == pytest.approx(0.40)

    def test_gate_does_not_mutate_its_input(self) -> None:
        hypotheses = [_hypothesis("Possibly a transient spike occurred", 0.40)]
        snapshot = copy.deepcopy(hypotheses)
        hypothesis_sanity_gate(_state(hypotheses))
        assert hypotheses == snapshot


# ── D4: convergence scorer purity ───────────────────────────────────────────


class TestConvergencePurity:
    def test_input_hypotheses_are_not_mutated(self) -> None:
        """DONE CHECK item 4."""
        hypotheses = [
            _hypothesis("Leading hypothesis under evaluation", 0.70, hypothesis_id="lead"),
            _hypothesis("Trailing hypothesis under evaluation", 0.20, hypothesis_id="trail"),
        ]
        snapshot = copy.deepcopy(hypotheses)
        node = make_update_convergence(persistence_required=True)
        node(_state(hypotheses))
        assert hypotheses == snapshot

    def test_returned_hypotheses_are_new_objects(self) -> None:
        hypotheses = [_hypothesis("Sole active hypothesis here", 0.90, hypothesis_id="only")]
        node = make_update_convergence(persistence_required=True)
        result = node(_state(hypotheses))
        assert result["hypotheses"][0] is not hypotheses[0]

    def test_repeated_scoring_is_deterministic(self) -> None:
        """In-place mutation made the result depend on how often it was called."""
        hypotheses = [
            _hypothesis("Leading hypothesis under evaluation", 0.70, hypothesis_id="lead"),
            _hypothesis("Trailing hypothesis under evaluation", 0.20, hypothesis_id="trail"),
        ]
        node = make_update_convergence(persistence_required=True)
        first = node(_state(copy.deepcopy(hypotheses)))
        second = node(_state(copy.deepcopy(hypotheses)))
        assert first["convergence_score"] == second["convergence_score"]
        assert first["hypotheses"] == second["hypotheses"]


# ── D5: anomaly score semantics ─────────────────────────────────────────────


class TestAnomalyScoreSemantics:
    def _adapt(self, **xai) -> Signal:
        payload = {
            "inputs_for_xai_model": {
                "signal_id": "00000000-0000-4000-8000-000000000001",
                "timestamp": _BASE.isoformat(),
                "signal_type": "generic",
                "entity": {"device": "10.0.0.1"},
                "features": ["dur"],
                "source": "unsw-threat-detector",
                **xai,
            }
        }
        return UNSWThreatAdapter().adapt(payload)

    def test_confident_normal_produces_low_anomaly_score(self) -> None:
        """DONE CHECK item 5, first half."""
        signal = self._adapt(
            signal_type="normal",
            anomaly_score=0.02,
            predicted_class_confidence=0.98,
            predictive_entropy=0.05,
        )
        assert signal.anomaly_score < 0.1
        assert signal.predicted_class_confidence > 0.9

    def test_confident_attack_produces_high_anomaly_score(self) -> None:
        """DONE CHECK item 5, second half."""
        signal = self._adapt(
            signal_type="generic",
            anomaly_score=0.97,
            predicted_class_confidence=0.97,
            predictive_entropy=0.08,
        )
        assert signal.anomaly_score > 0.9
        assert signal.predicted_class_confidence > 0.9

    def test_new_fields_survive_adaptation(self) -> None:
        signal = self._adapt(
            anomaly_score=0.6, predicted_class_confidence=0.7, predictive_entropy=0.4
        )
        assert signal.predicted_class_confidence == pytest.approx(0.7)
        assert signal.predictive_entropy == pytest.approx(0.4)

    def test_legacy_payload_without_new_fields_still_adapts(self) -> None:
        signal = self._adapt(anomaly_score=0.42, confidence=0.42)
        assert signal.anomaly_score == pytest.approx(0.42)
        assert signal.predicted_class_confidence is None

    def test_reasoning_engine_reads_the_corrected_field(self) -> None:
        engine = ReasoningEngine(clock_mode=ClockMode.SEPARATED)
        clock = ReplayClock()
        situation = Situation(clock=clock, clock_mode=ClockMode.SEPARATED)
        situation.attach_evidence(_signal_at(_BASE, anomaly_score=0.9, confidence=0.2))
        snapshot = engine.evaluate(situation)
        assert snapshot.mean_anomaly_score == pytest.approx(0.9)


# ── Ablation switches ───────────────────────────────────────────────────────


class TestAblationSwitches:
    def test_unknown_disabled_removes_the_hypothesis_entirely(self) -> None:
        llm = _FixedLLM(
            [{"description": "Coordinated probing across hosts", "confidence": 0.4}]
        )
        node = make_generate_hypotheses(lambda: llm, unknown_enabled=False)
        result = node(_state([]))
        ids = [h["hypothesis_id"] for h in result["hypotheses"]]
        assert UNKNOWN_HYPOTHESIS_ID not in ids

    def test_unknown_enabled_injects_the_hypothesis(self) -> None:
        llm = _FixedLLM(
            [{"description": "Coordinated probing across hosts", "confidence": 0.4}]
        )
        node = make_generate_hypotheses(lambda: llm, unknown_enabled=True)
        result = node(_state([]))
        ids = [h["hypothesis_id"] for h in result["hypotheses"]]
        assert UNKNOWN_HYPOTHESIS_ID in ids

    def test_asymmetric_decay_disabled_applies_delta_at_face_value(self) -> None:
        hypotheses = [_hypothesis("Declining hypothesis under quiet", 0.50)]
        state = _state(
            hypotheses,
            reasoning_snapshot=SituationReasoningSnapshot(
                situation_id="level2",
                evidence_count=6,
                event_rate=2.0,
                burst_detected=False,
                quiet_detected=True,
                confidence_level=0.0,
                trend=Trend.STABLE,
                source_diversity=2,
                mean_anomaly_score=0.5,
            ).model_dump(),
        )
        symmetric = make_evaluate_hypotheses(asymmetric_decay_enabled=False)(
            copy.deepcopy(state)
        )
        asymmetric = make_evaluate_hypotheses(asymmetric_decay_enabled=True)(
            copy.deepcopy(state)
        )
        assert symmetric["hypotheses"][0]["confidence"] == pytest.approx(0.40)
        assert asymmetric["hypotheses"][0]["confidence"] == pytest.approx(0.35)

    def test_persistence_disabled_allows_immediate_convergence(self) -> None:
        hypotheses = [
            _hypothesis("Dominant hypothesis with a clear lead", 0.95, hypothesis_id="lead"),
        ]
        without = make_update_convergence(persistence_required=False)(
            _state(copy.deepcopy(hypotheses))
        )
        with_persistence = make_update_convergence(persistence_required=True)(
            _state(copy.deepcopy(hypotheses))
        )
        assert without["convergence_score"] >= 0.8
        assert with_persistence["convergence_score"] < 0.8

    def test_sanity_gate_absent_from_graph_when_disabled(self) -> None:
        llm = _FixedLLM([{"description": "Coordinated probing across hosts", "confidence": 0.4}])
        disabled = build_reasoning_graph(
            lambda: llm, EpistemicControls(sanity_gate_enabled=False)
        )
        enabled = build_reasoning_graph(
            lambda: llm, EpistemicControls(sanity_gate_enabled=True)
        )
        assert "hypothesis_sanity_gate" not in disabled.get_graph().nodes
        assert "hypothesis_sanity_gate" in enabled.get_graph().nodes

    def test_inertia_absent_from_graph_when_cap_is_infinite(self) -> None:
        llm = _FixedLLM([{"description": "Coordinated probing across hosts", "confidence": 0.4}])
        disabled = build_reasoning_graph(
            lambda: llm, EpistemicControls(max_confidence_delta=float("inf"))
        )
        enabled = build_reasoning_graph(
            lambda: llm, EpistemicControls(max_confidence_delta=0.15)
        )
        assert "apply_belief_inertia" not in disabled.get_graph().nodes
        assert "apply_belief_inertia" in enabled.get_graph().nodes

    def test_all_sixteen_configurations_build(self) -> None:
        """Level 9 runs a 2^4 factorial, so every corner must compile."""
        llm = _FixedLLM([{"description": "Coordinated probing across hosts", "confidence": 0.4}])
        built = 0
        for unknown in (True, False):
            for gate in (True, False):
                for decay in (True, False):
                    for persistence in (True, False):
                        controls = EpistemicControls(
                            unknown_hypothesis_enabled=unknown,
                            sanity_gate_enabled=gate,
                            asymmetric_decay_enabled=decay,
                            persistence_required=persistence,
                        )
                        assert build_reasoning_graph(lambda: llm, controls) is not None
                        built += 1
        assert built == 16


class _FixedLLM:
    """Minimal stand-in that returns a fixed JSON hypothesis array."""

    def __init__(self, hypotheses: list[dict]) -> None:
        import json
        from types import SimpleNamespace

        self._response = SimpleNamespace(content=json.dumps(hypotheses))

    def invoke(self, prompt: str):
        """Return the fixed response regardless of the prompt."""
        return self._response


# ── Configuration defaults ──────────────────────────────────────────────────


class TestConfigurationDefaults:
    def test_all_six_switches_have_the_stated_defaults(self) -> None:
        """DONE CHECK item 7."""
        from enigma_reason.config import Settings

        fresh = Settings()
        assert fresh.clock_mode is ClockMode.SEPARATED
        assert fresh.unknown_hypothesis_enabled is True
        assert fresh.sanity_gate_enabled is True
        assert fresh.asymmetric_decay_enabled is True
        assert fresh.persistence_required is True
        assert fresh.max_confidence_delta == pytest.approx(0.15)

    def test_switches_read_from_environment(self, monkeypatch) -> None:
        from enigma_reason.config import Settings

        monkeypatch.setenv("ENIGMA_CLOCK_MODE", "conflated")
        monkeypatch.setenv("ENIGMA_UNKNOWN_HYPOTHESIS_ENABLED", "false")
        monkeypatch.setenv("ENIGMA_SANITY_GATE_ENABLED", "false")
        monkeypatch.setenv("ENIGMA_ASYMMETRIC_DECAY_ENABLED", "false")
        monkeypatch.setenv("ENIGMA_PERSISTENCE_REQUIRED", "false")
        monkeypatch.setenv("ENIGMA_MAX_CONFIDENCE_DELTA", "0.42")

        overridden = Settings()
        assert overridden.clock_mode is ClockMode.CONFLATED
        assert overridden.unknown_hypothesis_enabled is False
        assert overridden.sanity_gate_enabled is False
        assert overridden.asymmetric_decay_enabled is False
        assert overridden.persistence_required is False
        assert overridden.max_confidence_delta == pytest.approx(0.42)
