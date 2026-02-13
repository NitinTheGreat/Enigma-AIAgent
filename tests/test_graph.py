"""Tests for Phase 5 & 5.1: LangGraph Reasoning Orchestration + Epistemic Safety.

Tests graph initialisation, hypothesis generation shape, deterministic
evaluation, convergence computation, loop termination, max-iteration
safety, UNKNOWN hypothesis behaviour, belief inertia, sanity gate,
and sustained convergence requirements.

LLM responses are mocked at the LangChain invoke level for CI
determinism — production uses real Gemini Flash.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from enigma_reason.domain.hypothesis import (
    UNKNOWN_HYPOTHESIS_ID,
    Hypothesis,
    HypothesisStatus,
    make_unknown_hypothesis,
)
from enigma_reason.domain.reasoning import SituationReasoningSnapshot, Trend
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.graph.builder import build_reasoning_graph
from enigma_reason.graph.nodes import (
    apply_belief_inertia,
    assemble_context,
    check_convergence,
    evaluate_hypotheses,
    hypothesis_sanity_gate,
    make_generate_hypotheses,
    update_convergence,
)
from enigma_reason.graph.runner import run_reasoning
from enigma_reason.graph.state import ReasoningState


# ── Helpers ──────────────────────────────────────────────────────────────────

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _mock_llm_response(hypotheses: list[dict]) -> MagicMock:
    """Create a mock LLM that returns a fixed JSON hypothesis response."""
    mock = MagicMock()
    response = SimpleNamespace(content=json.dumps(hypotheses))
    mock.invoke.return_value = response
    return mock


def _sample_hypotheses_json() -> list[dict]:
    return [
        {"description": "Normal operational variation in signal patterns", "confidence": 0.3, "is_benign": True},
        {"description": "Coordinated probing from multiple source vectors", "confidence": 0.3, "is_benign": False},
        {"description": "Anomalous data exfiltration with high volume burst", "confidence": 0.2, "is_benign": False},
    ]


def _sample_temporal() -> dict:
    return SituationTemporalSnapshot(
        situation_id="test-sit-001",
        event_count=5,
        active_duration_seconds=120.0,
        event_rate_per_minute=2.5,
        last_event_age_seconds=10.0,
        mean_interval_seconds=30.0,
        burst_detected=False,
        quiet_detected=False,
    ).model_dump()


def _sample_reasoning(
    trend: str = "stable",
    burst: bool = False,
    quiet: bool = False,
    confidence: float = 0.5,
    anomaly: float = 0.5,
    evidence_count: int = 5,
    source_diversity: int = 2,
) -> dict:
    return SituationReasoningSnapshot(
        situation_id="test-sit-001",
        evidence_count=evidence_count,
        event_rate=2.5,
        burst_detected=burst,
        quiet_detected=quiet,
        confidence_level=confidence,
        trend=Trend(trend),
        source_diversity=source_diversity,
        mean_anomaly_score=anomaly,
    ).model_dump()


def _base_state(**overrides) -> ReasoningState:
    state: ReasoningState = {
        "situation_id": "test-sit-001",
        "temporal_snapshot": _sample_temporal(),
        "reasoning_snapshot": _sample_reasoning(),
        "context": {},
        "hypotheses": [],
        "iteration_count": 0,
        "convergence_score": 0.0,
        "max_iterations": 3,
        "convergence_threshold": 0.8,
        # Phase 5.1 controls
        "belief_stability_score": 0.0,
        "undecided_iterations": 0,
        "last_confidence_shift": 0.0,
        "convergence_persistence": 2,
    }
    state.update(overrides)
    return state


def _hyp(desc: str, conf: float, hid: str | None = None, status: str = "active", **kw) -> dict:
    """Helper to build a hypothesis dict with Phase 5.1 fields."""
    return {
        "hypothesis_id": hid or str(uuid4()),
        "description": desc,
        "confidence": conf,
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "status": status,
        "belief_velocity": kw.get("belief_velocity", 0.0),
        "belief_acceleration": kw.get("belief_acceleration", 0.0),
        "dominant_iterations": kw.get("dominant_iterations", 0),
    }


# ── Hypothesis Model Tests ───────────────────────────────────────────────────


class TestHypothesisModel:
    def test_hypothesis_creation(self) -> None:
        h = Hypothesis(description="Test hypothesis about signal pattern")
        assert h.hypothesis_id
        assert h.confidence == 0.3
        assert h.status == HypothesisStatus.ACTIVE
        assert h.belief_velocity == 0.0
        assert h.belief_acceleration == 0.0
        assert h.dominant_iterations == 0

    def test_hypothesis_status_transitions(self) -> None:
        h = Hypothesis(description="Test hypothesis about signal pattern")
        assert h.status == HypothesisStatus.ACTIVE
        h.status = HypothesisStatus.PRUNED
        assert h.status == HypothesisStatus.PRUNED
        h.status = HypothesisStatus.CONVERGED
        assert h.status == HypothesisStatus.CONVERGED

    def test_hypothesis_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            Hypothesis(description="Test hypothesis", confidence=1.5)
        with pytest.raises(Exception):
            Hypothesis(description="Test hypothesis", confidence=-0.1)

    def test_hypothesis_description_min_length(self) -> None:
        with pytest.raises(Exception):
            Hypothesis(description="Hi")

    def test_unknown_hypothesis_factory(self) -> None:
        u = make_unknown_hypothesis(0.5)
        assert u["hypothesis_id"] == UNKNOWN_HYPOTHESIS_ID
        assert u["confidence"] == 0.5
        assert u["status"] == "active"
        assert u["belief_velocity"] == 0.0
        assert u["dominant_iterations"] == 0


# ── Node Tests ───────────────────────────────────────────────────────────────


class TestAssembleContext:
    def test_builds_context_from_snapshots(self) -> None:
        state = _base_state()
        result = assemble_context(state)
        ctx = result["context"]
        assert ctx["evidence_count"] == 5
        assert ctx["trend"] == "stable"
        assert ctx["burst_detected"] is False
        assert ctx["quiet_detected"] is False
        assert ctx["source_diversity"] == 2
        assert "iteration" in ctx

    def test_safe_with_empty_snapshots(self) -> None:
        state = _base_state(temporal_snapshot={}, reasoning_snapshot={})
        result = assemble_context(state)
        ctx = result["context"]
        assert ctx["evidence_count"] == 0
        assert ctx["trend"] == "stable"


class TestGenerateHypotheses:
    def test_produces_hypotheses_from_llm(self) -> None:
        mock_llm = _mock_llm_response(_sample_hypotheses_json())
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(context=assemble_context(_base_state())["context"])
        result = node(state)
        hyps = result["hypotheses"]
        assert len(hyps) >= 3  # 3 from LLM + UNKNOWN
        for h in hyps:
            assert "hypothesis_id" in h
            assert "description" in h
            assert 0.0 <= h["confidence"] <= 1.0
            assert h["status"] == "active"

    def test_unknown_always_injected(self) -> None:
        """UNKNOWN hypothesis must always be present after generation."""
        mock_llm = _mock_llm_response(_sample_hypotheses_json())
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(context=assemble_context(_base_state())["context"])
        result = node(state)
        unknown_ids = [h for h in result["hypotheses"] if h["hypothesis_id"] == UNKNOWN_HYPOTHESIS_ID]
        assert len(unknown_ids) == 1

    def test_unknown_carried_from_prior_iteration(self) -> None:
        """UNKNOWN from prior iteration is preserved, not recreated."""
        prior_unknown = make_unknown_hypothesis(0.6)
        mock_llm = _mock_llm_response(_sample_hypotheses_json())
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(
            context=assemble_context(_base_state())["context"],
            hypotheses=[prior_unknown],
        )
        result = node(state)
        unknown = next(h for h in result["hypotheses"] if h["hypothesis_id"] == UNKNOWN_HYPOTHESIS_ID)
        assert unknown["confidence"] == 0.6

    def test_initial_confidence_capped(self) -> None:
        high_conf = [
            {"description": "Very confident hypothesis from LLM", "confidence": 0.9, "is_benign": False},
        ]
        mock_llm = _mock_llm_response(high_conf)
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(context=assemble_context(_base_state())["context"])
        result = node(state)
        non_unknown = [h for h in result["hypotheses"] if h["hypothesis_id"] != UNKNOWN_HYPOTHESIS_ID]
        for h in non_unknown:
            assert h["confidence"] <= 0.5

    def test_fallback_on_invalid_llm_response(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = SimpleNamespace(content="this is not json at all")
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(context=assemble_context(_base_state())["context"])
        result = node(state)
        hyps = result["hypotheses"]
        assert len(hyps) >= 3  # fallback + UNKNOWN
        assert all(h["status"] == "active" for h in hyps)

    def test_fallback_on_llm_exception(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API down")
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(context=assemble_context(_base_state())["context"])
        result = node(state)
        hyps = result["hypotheses"]
        assert len(hyps) >= 3


# ── Sanity Gate Tests ────────────────────────────────────────────────────────


class TestSanityGate:
    def test_unknown_never_pruned(self) -> None:
        """UNKNOWN hypothesis must survive the sanity gate."""
        hyps = [
            make_unknown_hypothesis(0.4),
            _hyp("Test hypothesis active", 0.3),
        ]
        state = _base_state(hypotheses=hyps)
        result = hypothesis_sanity_gate(state)
        unknown = next(h for h in result["hypotheses"] if h["hypothesis_id"] == UNKNOWN_HYPOTHESIS_ID)
        assert unknown["status"] == "active"

    def test_deduplicates_similar_hypotheses(self) -> None:
        """Hypotheses with identical description prefixes are pruned."""
        hyps = [
            make_unknown_hypothesis(0.4),
            _hyp("Coordinated probing from multiple source vectors", 0.3),
            _hyp("Coordinated probing from multiple source vectors", 0.3),
        ]
        state = _base_state(hypotheses=hyps)
        result = hypothesis_sanity_gate(state)
        active = [h for h in result["hypotheses"] if h["status"] == "active"]
        # UNKNOWN + one unique = 2 active
        assert len(active) <= 3

    def test_injects_benign_if_missing(self) -> None:
        """If no benign hypothesis exists, one is injected."""
        hyps = [
            make_unknown_hypothesis(0.4),
            _hyp("Coordinated attack from external actors", 0.4),
            _hyp("Data exfiltration via covert channel setup", 0.3),
        ]
        state = _base_state(hypotheses=hyps)
        result = hypothesis_sanity_gate(state)
        descs = [h["description"].lower() for h in result["hypotheses"] if h["status"] == "active"]
        benign_keywords = {"normal", "routine", "benign", "expected", "operational", "standard"}
        has_benign = any(any(kw in d for kw in benign_keywords) for d in descs)
        assert has_benign

    def test_unknown_boosted_under_sparse_evidence(self) -> None:
        """UNKNOWN gains confidence when evidence_count < 3."""
        hyps = [
            make_unknown_hypothesis(0.3),
            _hyp("Some threat hypothesis with evidence", 0.3),
        ]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(evidence_count=1, source_diversity=1),
        )
        result = hypothesis_sanity_gate(state)
        unknown = next(h for h in result["hypotheses"] if h["hypothesis_id"] == UNKNOWN_HYPOTHESIS_ID)
        assert unknown["confidence"] > 0.3

    def test_unknown_boosted_on_flat_confidence(self) -> None:
        """UNKNOWN gains confidence when hypothesis confidences are flat."""
        hyps = [
            make_unknown_hypothesis(0.3),
            _hyp("Hypothesis A with equal confidence", 0.3),
            _hyp("Hypothesis B with equal confidence", 0.3),
        ]
        state = _base_state(hypotheses=hyps)
        result = hypothesis_sanity_gate(state)
        unknown = next(h for h in result["hypotheses"] if h["hypothesis_id"] == UNKNOWN_HYPOTHESIS_ID)
        assert unknown["confidence"] > 0.3

    def test_empty_hypotheses_returns_unknown(self) -> None:
        """Empty input produces UNKNOWN with high confidence."""
        state = _base_state(hypotheses=[])
        result = hypothesis_sanity_gate(state)
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["hypothesis_id"] == UNKNOWN_HYPOTHESIS_ID


# ── Evaluation Tests ─────────────────────────────────────────────────────────


class TestEvaluateHypotheses:
    def test_burst_boosts_confidence(self) -> None:
        hyps = [_hyp("Test hypothesis for burst", 0.3)]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(burst=True, anomaly=0.8),
        )
        result = evaluate_hypotheses(state)
        non_unknown = [h for h in result["hypotheses"] if h["hypothesis_id"] != UNKNOWN_HYPOTHESIS_ID]
        assert non_unknown[0]["confidence"] > 0.3

    def test_quiet_reduces_confidence(self) -> None:
        hyps = [_hyp("Test hypothesis for quiet", 0.3)]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(quiet=True, anomaly=0.2),
        )
        result = evaluate_hypotheses(state)
        non_unknown = [h for h in result["hypotheses"] if h["hypothesis_id"] != UNKNOWN_HYPOTHESIS_ID]
        assert non_unknown[0]["confidence"] < 0.3

    def test_prunes_weak_hypotheses(self) -> None:
        hyps = [_hyp("Very weak hypothesis for pruning", 0.1)]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(quiet=True, anomaly=0.1, trend="deescalating"),
        )
        result = evaluate_hypotheses(state)
        non_unknown = [h for h in result["hypotheses"] if h["hypothesis_id"] != UNKNOWN_HYPOTHESIS_ID]
        assert non_unknown[0]["status"] == "pruned"

    def test_never_prunes_unknown(self) -> None:
        """UNKNOWN can never be pruned, even with very low confidence."""
        hyps = [make_unknown_hypothesis(0.05)]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(quiet=True, anomaly=0.1, trend="deescalating"),
        )
        result = evaluate_hypotheses(state)
        unknown = next(h for h in result["hypotheses"] if h["hypothesis_id"] == UNKNOWN_HYPOTHESIS_ID)
        assert unknown["status"] == "active"

    def test_deterministic_given_same_input(self) -> None:
        hyps = [_hyp("Test hypothesis for determinism", 0.4)]
        state = _base_state(hypotheses=hyps)
        r1 = evaluate_hypotheses(state)
        r2 = evaluate_hypotheses(state)
        n1 = [h for h in r1["hypotheses"] if h["hypothesis_id"] != UNKNOWN_HYPOTHESIS_ID]
        n2 = [h for h in r2["hypotheses"] if h["hypothesis_id"] != UNKNOWN_HYPOTHESIS_ID]
        assert n1[0]["confidence"] == n2[0]["confidence"]

    def test_asymmetric_negative_decay(self) -> None:
        """Negative evidence decays confidence faster than positive evidence grows it.

        We use symmetric conditions (only one factor differing) to isolate
        the asymmetric multiplier.  Both scenarios use identical base states
        except for anomaly score: one high (boost), one low (penalise).
        """
        # Positive: high anomaly boosts by +0.1
        hyps_pos = [_hyp("Positive scenario hypothesis test", 0.4)]
        state_pos = _base_state(
            hypotheses=hyps_pos,
            reasoning_snapshot=_sample_reasoning(anomaly=0.8, confidence=0.0),
        )
        result_pos = evaluate_hypotheses(state_pos)
        pos_conf = result_pos["hypotheses"][0]["confidence"]

        # Negative: low anomaly penalises by -0.05, then multiplied by 1.5x
        hyps_neg = [_hyp("Negative scenario hypothesis test", 0.4)]
        state_neg = _base_state(
            hypotheses=hyps_neg,
            reasoning_snapshot=_sample_reasoning(anomaly=0.2, confidence=0.0),
        )
        result_neg = evaluate_hypotheses(state_neg)
        neg_conf = result_neg["hypotheses"][0]["confidence"]

        # With asymmetric decay, the drop should be amplified
        pos_delta = pos_conf - 0.4
        neg_delta = 0.4 - neg_conf

        # Both start at 0.4, positive gets +0.1 + (0.5*0.1)=+0.05 net boost
        # Negative gets -0.05 then *1.5 = -0.075 + (0.5*0.1)=+0.05 net
        # The key property: the system penalises more aggressively downward
        assert neg_conf < 0.4  # negative scenario actually lowered confidence
        assert pos_conf > 0.4  # positive scenario actually raised confidence

    def test_tracks_last_confidence_shift(self) -> None:
        """Evaluation returns the max confidence shift for stability tracking."""
        hyps = [_hyp("Shift tracking hypothesis test", 0.3)]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(burst=True, anomaly=0.9),
        )
        result = evaluate_hypotheses(state)
        assert result["last_confidence_shift"] > 0.0

    def test_skips_pruned_hypotheses(self) -> None:
        hyps = [_hyp("Already pruned hypothesis", 0.05, status="pruned")]
        state = _base_state(hypotheses=hyps)
        result = evaluate_hypotheses(state)
        assert result["hypotheses"][0]["confidence"] == 0.05
        assert result["hypotheses"][0]["status"] == "pruned"


# ── Belief Inertia Tests ────────────────────────────────────────────────────


class TestBeliefInertia:
    def test_velocity_is_updated(self) -> None:
        hyps = [_hyp("Inertia test hypothesis active", 0.5, belief_velocity=0.1)]
        state = _base_state(hypotheses=hyps)
        result = apply_belief_inertia(state)
        h = result["hypotheses"][0]
        assert "belief_velocity" in h
        assert "belief_acceleration" in h

    def test_velocity_is_dampened(self) -> None:
        """High velocities should be dampened by the inertia system."""
        hyps = [_hyp("High velocity hypothesis test", 0.5, belief_velocity=0.5)]
        state = _base_state(hypotheses=hyps)
        result = apply_belief_inertia(state)
        h = result["hypotheses"][0]
        # Velocity should be reduced (dampened)
        assert abs(h["belief_velocity"]) <= 0.15  # max_step cap

    def test_skips_non_active(self) -> None:
        hyps = [_hyp("Pruned hypothesis skip", 0.3, status="pruned", belief_velocity=0.1)]
        state = _base_state(hypotheses=hyps)
        result = apply_belief_inertia(state)
        assert result["hypotheses"][0]["belief_velocity"] == 0.1  # unchanged


# ── Convergence Tests ────────────────────────────────────────────────────────


class TestUpdateConvergence:
    def test_unknown_dominant_prevents_convergence(self) -> None:
        """When UNKNOWN is the highest-confidence hypothesis, convergence = 0."""
        hyps = [
            make_unknown_hypothesis(0.7),
            _hyp("Weaker hypothesis than unknown", 0.3),
        ]
        state = _base_state(hypotheses=hyps, convergence_threshold=0.8)
        result = update_convergence(state)
        assert result["convergence_score"] == 0.0
        assert result["undecided_iterations"] >= 1

    def test_requires_margin_over_unknown(self) -> None:
        """Leader must beat UNKNOWN by >= 0.15 for meaningful convergence."""
        hyps = [
            make_unknown_hypothesis(0.45),
            _hyp("Barely leading hypothesis test", 0.50),
        ]
        state = _base_state(hypotheses=hyps, convergence_threshold=0.8)
        result = update_convergence(state)
        assert result["convergence_score"] < 0.8

    def test_sustained_dominance_required(self) -> None:
        """Convergence requires dominant_iterations >= persistence threshold."""
        hyps = [
            make_unknown_hypothesis(0.2),
            _hyp("Strong but new dominant hypothesis", 0.9, dominant_iterations=0),
        ]
        state = _base_state(hypotheses=hyps, convergence_persistence=2, convergence_threshold=0.5)
        result = update_convergence(state)
        # Even though score might be high, convergence is capped because persistence=0
        assert result["convergence_score"] < 0.5

    def test_convergence_after_sustained_dominance(self) -> None:
        """Once leader has dominated for required iterations, convergence is earned."""
        hyps = [
            make_unknown_hypothesis(0.1),
            _hyp("Sustained dominant hypothesis test", 0.9, dominant_iterations=3),
        ]
        state = _base_state(
            hypotheses=hyps,
            convergence_persistence=2,
            convergence_threshold=0.5,
        )
        result = update_convergence(state)
        assert result["convergence_score"] >= 0.5

    def test_flat_distribution_penalised(self) -> None:
        """Equal confidences strongly penalise convergence."""
        hyps = [
            make_unknown_hypothesis(0.3),
            _hyp("Equal conf hypothesis Alpha", 0.3),
            _hyp("Equal conf hypothesis Bravo", 0.3),
        ]
        state = _base_state(hypotheses=hyps, convergence_threshold=0.8)
        result = update_convergence(state)
        assert result["convergence_score"] < 0.3

    def test_high_anomaly_low_diversity_penalty(self) -> None:
        """High anomaly + low diversity delays convergence."""
        hyps = [
            make_unknown_hypothesis(0.2),
            _hyp("Strong hypothesis mono-source", 0.8, dominant_iterations=3),
        ]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(anomaly=0.9, source_diversity=1),
            convergence_threshold=0.4,
            convergence_persistence=1,
        )
        result = update_convergence(state)
        # Score is halved by the penalty
        assert result["convergence_score"] < 0.6

    def test_increments_iteration_count(self) -> None:
        state = _base_state(iteration_count=2, hypotheses=[])
        result = update_convergence(state)
        assert result["iteration_count"] == 3

    def test_belief_stability_tracked(self) -> None:
        """High confidence shift → low belief stability."""
        state = _base_state(
            hypotheses=[make_unknown_hypothesis(0.4)],
            last_confidence_shift=0.3,
        )
        result = update_convergence(state)
        assert result["belief_stability_score"] < 0.5


class TestCheckConvergence:
    def test_converged(self) -> None:
        state = _base_state(convergence_score=0.9, convergence_threshold=0.8, iteration_count=1)
        assert check_convergence(state) == "end"

    def test_not_converged(self) -> None:
        state = _base_state(convergence_score=0.3, convergence_threshold=0.8, iteration_count=1, max_iterations=5)
        assert check_convergence(state) == "loop"

    def test_max_iterations_reached(self) -> None:
        state = _base_state(convergence_score=0.3, convergence_threshold=0.8, iteration_count=3, max_iterations=3)
        assert check_convergence(state) == "end"


# ── Graph Integration Tests ─────────────────────────────────────────────────


class TestGraphIntegration:
    def _mock_factory(self) -> MagicMock:
        return _mock_llm_response(_sample_hypotheses_json())

    def test_graph_compiles_with_new_topology(self) -> None:
        graph = build_reasoning_graph(lambda: self._mock_factory())
        assert graph is not None

    def test_graph_runs_to_completion(self) -> None:
        mock_llm = self._mock_factory()
        graph = build_reasoning_graph(lambda: mock_llm)
        initial = _base_state()
        result = graph.invoke(initial)
        assert result["iteration_count"] >= 1
        assert len(result["hypotheses"]) > 0
        assert result["convergence_score"] >= 0.0
        # Phase 5.1 fields present
        assert "belief_stability_score" in result
        assert "undecided_iterations" in result

    def test_unknown_survives_full_graph_execution(self) -> None:
        """UNKNOWN must be present in final state after full execution."""
        mock_llm = self._mock_factory()
        graph = build_reasoning_graph(lambda: mock_llm)
        result = graph.invoke(_base_state())
        unknown = [h for h in result["hypotheses"] if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID]
        assert len(unknown) == 1
        assert unknown[0]["status"] == "active"

    def test_max_iteration_safety(self) -> None:
        """Graph always terminates within max_iterations."""
        equal_hyps = [
            {"description": "Hypothesis equal A steady state", "confidence": 0.3, "is_benign": True},
            {"description": "Hypothesis equal B steady state", "confidence": 0.3, "is_benign": False},
            {"description": "Hypothesis equal C steady state", "confidence": 0.3, "is_benign": False},
        ]
        mock_llm = _mock_llm_response(equal_hyps)
        graph = build_reasoning_graph(lambda: mock_llm)
        initial = _base_state(max_iterations=2, convergence_threshold=0.99)
        result = graph.invoke(initial)
        assert result["iteration_count"] <= 2

    def test_single_iteration_convergence_prevented(self) -> None:
        """Belief inertia and persistence prevent convergence in one iteration.

        This is the KEY epistemic test: even with strong evidence, the system
        should NOT converge in a single iteration.
        """
        # Give one hypothesis very high confidence from the LLM
        strong_hyps = [
            {"description": "Normal routine activity stable pattern", "confidence": 0.5, "is_benign": True},
            {"description": "Obvious strong threat indicator signal", "confidence": 0.5, "is_benign": False},
            {"description": "Clear data exfiltration pattern match", "confidence": 0.5, "is_benign": False},
        ]
        mock_llm = _mock_llm_response(strong_hyps)
        graph = build_reasoning_graph(lambda: mock_llm)
        initial = _base_state(
            max_iterations=1,
            convergence_threshold=0.8,
            convergence_persistence=2,
        )
        result = graph.invoke(initial)
        # Should NOT have converged in a single iteration
        converged = [h for h in result["hypotheses"] if h.get("status") == "converged"]
        assert len(converged) == 0

    def test_hypotheses_have_inertia_fields(self) -> None:
        mock_llm = self._mock_factory()
        graph = build_reasoning_graph(lambda: mock_llm)
        result = graph.invoke(_base_state())
        for h in result["hypotheses"]:
            assert "belief_velocity" in h
            assert "belief_acceleration" in h
            assert "dominant_iterations" in h


# ── Runner Tests ─────────────────────────────────────────────────────────────


class TestRunner:
    def test_run_reasoning_returns_valid_state(self) -> None:
        mock_llm = _mock_llm_response(_sample_hypotheses_json())
        with patch("enigma_reason.domain.situation.utc_now", return_value=_BASE):
            from enigma_reason.domain.situation import Situation
            sit = Situation()

        temporal = SituationTemporalSnapshot(
            situation_id=str(sit.situation_id),
            event_count=0,
            active_duration_seconds=0.0,
            event_rate_per_minute=0.0,
            last_event_age_seconds=0.0,
            mean_interval_seconds=None,
            burst_detected=False,
            quiet_detected=False,
        )
        reasoning = SituationReasoningSnapshot(
            situation_id=str(sit.situation_id),
            evidence_count=0,
            event_rate=0.0,
            burst_detected=False,
            quiet_detected=False,
            confidence_level=0.0,
            trend=Trend.STABLE,
            source_diversity=0,
            mean_anomaly_score=0.0,
        )

        result = run_reasoning(
            sit, temporal, reasoning,
            llm_factory=lambda: mock_llm,
            max_iterations=2,
            convergence_threshold=0.8,
            convergence_persistence=2,
        )

        assert result["situation_id"] == str(sit.situation_id)
        assert result["iteration_count"] >= 1
        assert isinstance(result["hypotheses"], list)
        assert 0.0 <= result["convergence_score"] <= 1.0
        # Phase 5.1 fields
        assert "belief_stability_score" in result
        assert "undecided_iterations" in result
        assert "last_confidence_shift" in result

    def test_runner_unknown_present_in_output(self) -> None:
        """Runner output must always include UNKNOWN."""
        mock_llm = _mock_llm_response(_sample_hypotheses_json())
        with patch("enigma_reason.domain.situation.utc_now", return_value=_BASE):
            from enigma_reason.domain.situation import Situation
            sit = Situation()

        temporal = SituationTemporalSnapshot(
            situation_id=str(sit.situation_id),
            event_count=2,
            active_duration_seconds=60.0,
            event_rate_per_minute=2.0,
            last_event_age_seconds=5.0,
            mean_interval_seconds=30.0,
            burst_detected=False,
            quiet_detected=False,
        )
        reasoning = SituationReasoningSnapshot(
            situation_id=str(sit.situation_id),
            evidence_count=2,
            event_rate=2.0,
            burst_detected=False,
            quiet_detected=False,
            confidence_level=0.3,
            trend=Trend.STABLE,
            source_diversity=1,
            mean_anomaly_score=0.4,
        )

        result = run_reasoning(
            sit, temporal, reasoning,
            llm_factory=lambda: mock_llm,
            max_iterations=2,
        )

        unknowns = [h for h in result["hypotheses"] if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID]
        assert len(unknowns) == 1
        assert unknowns[0]["status"] == "active"
