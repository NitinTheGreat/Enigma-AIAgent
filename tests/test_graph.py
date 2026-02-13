"""Tests for Phase 5: LangGraph Reasoning Orchestration.

Tests graph initialisation, hypothesis generation shape, deterministic
evaluation, convergence computation, loop termination, and max-iteration
safety.  LLM responses are mocked at the LangChain invoke level for CI
determinism — production uses real Gemini Flash.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from enigma_reason.domain.hypothesis import Hypothesis, HypothesisStatus
from enigma_reason.domain.reasoning import SituationReasoningSnapshot, Trend
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.graph.builder import build_reasoning_graph
from enigma_reason.graph.nodes import (
    assemble_context,
    check_convergence,
    evaluate_hypotheses,
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
) -> dict:
    return SituationReasoningSnapshot(
        situation_id="test-sit-001",
        evidence_count=5,
        event_rate=2.5,
        burst_detected=burst,
        quiet_detected=quiet,
        confidence_level=confidence,
        trend=Trend(trend),
        source_diversity=2,
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
    }
    state.update(overrides)
    return state


# ── Hypothesis Model Tests ───────────────────────────────────────────────────


class TestHypothesisModel:
    def test_hypothesis_creation(self) -> None:
        h = Hypothesis(description="Test hypothesis about signal pattern")
        assert h.hypothesis_id  # auto-generated
        assert h.confidence == 0.3  # default
        assert h.status == HypothesisStatus.ACTIVE
        assert h.supporting_evidence_ids == []
        assert h.contradicting_evidence_ids == []

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
            Hypothesis(description="Hi")  # too short


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
        assert len(hyps) == 3
        for h in hyps:
            assert "hypothesis_id" in h
            assert "description" in h
            assert 0.0 <= h["confidence"] <= 1.0
            assert h["status"] == "active"

    def test_initial_confidence_capped(self) -> None:
        """LLM cannot set initial confidence above 0.5."""
        high_conf = [
            {"description": "Very confident hypothesis from LLM", "confidence": 0.9, "is_benign": False},
        ]
        mock_llm = _mock_llm_response(high_conf)
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(context=assemble_context(_base_state())["context"])
        result = node(state)
        for h in result["hypotheses"]:
            assert h["confidence"] <= 0.5

    def test_fallback_on_invalid_llm_response(self) -> None:
        """If LLM returns garbage, fallback hypotheses are used."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = SimpleNamespace(content="this is not json at all")
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(context=assemble_context(_base_state())["context"])
        result = node(state)
        hyps = result["hypotheses"]
        assert len(hyps) == 3
        assert all(h["status"] == "active" for h in hyps)

    def test_fallback_on_llm_exception(self) -> None:
        """If LLM invocation raises, fallback hypotheses are used."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API down")
        node = make_generate_hypotheses(lambda: mock_llm)
        state = _base_state(context=assemble_context(_base_state())["context"])
        result = node(state)
        hyps = result["hypotheses"]
        assert len(hyps) == 3


class TestEvaluateHypotheses:
    def test_burst_boosts_confidence(self) -> None:
        hyps = [{"hypothesis_id": "h1", "description": "Test hyp", "confidence": 0.3, "status": "active",
                 "supporting_evidence_ids": [], "contradicting_evidence_ids": []}]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(burst=True, anomaly=0.8),
        )
        result = evaluate_hypotheses(state)
        assert result["hypotheses"][0]["confidence"] > 0.3

    def test_quiet_reduces_confidence(self) -> None:
        hyps = [{"hypothesis_id": "h1", "description": "Test hyp", "confidence": 0.3, "status": "active",
                 "supporting_evidence_ids": [], "contradicting_evidence_ids": []}]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(quiet=True, anomaly=0.2),
        )
        result = evaluate_hypotheses(state)
        assert result["hypotheses"][0]["confidence"] < 0.3

    def test_prunes_weak_hypotheses(self) -> None:
        hyps = [{"hypothesis_id": "h1", "description": "Very weak hypothesis", "confidence": 0.1, "status": "active",
                 "supporting_evidence_ids": [], "contradicting_evidence_ids": []}]
        state = _base_state(
            hypotheses=hyps,
            reasoning_snapshot=_sample_reasoning(quiet=True, anomaly=0.1, trend="deescalating"),
        )
        result = evaluate_hypotheses(state)
        assert result["hypotheses"][0]["status"] == "pruned"

    def test_deterministic_given_same_input(self) -> None:
        hyps = [{"hypothesis_id": "h1", "description": "Test hypothesis", "confidence": 0.4, "status": "active",
                 "supporting_evidence_ids": [], "contradicting_evidence_ids": []}]
        state = _base_state(hypotheses=hyps)
        r1 = evaluate_hypotheses(state)
        r2 = evaluate_hypotheses(state)
        assert r1["hypotheses"][0]["confidence"] == r2["hypotheses"][0]["confidence"]

    def test_skips_pruned_hypotheses(self) -> None:
        hyps = [{"hypothesis_id": "h1", "description": "Already pruned", "confidence": 0.05, "status": "pruned",
                 "supporting_evidence_ids": [], "contradicting_evidence_ids": []}]
        state = _base_state(hypotheses=hyps)
        result = evaluate_hypotheses(state)
        assert result["hypotheses"][0]["confidence"] == 0.05
        assert result["hypotheses"][0]["status"] == "pruned"


class TestUpdateConvergence:
    def test_single_active_uses_confidence(self) -> None:
        hyps = [
            {"hypothesis_id": "h1", "description": "Dominant", "confidence": 0.8, "status": "active",
             "supporting_evidence_ids": [], "contradicting_evidence_ids": []},
        ]
        state = _base_state(hypotheses=hyps, convergence_threshold=0.8)
        result = update_convergence(state)
        assert result["convergence_score"] >= 0.8
        assert result["iteration_count"] == 1

    def test_equal_confidence_low_convergence(self) -> None:
        hyps = [
            {"hypothesis_id": "h1", "description": "Hypothesis A", "confidence": 0.4, "status": "active",
             "supporting_evidence_ids": [], "contradicting_evidence_ids": []},
            {"hypothesis_id": "h2", "description": "Hypothesis B", "confidence": 0.4, "status": "active",
             "supporting_evidence_ids": [], "contradicting_evidence_ids": []},
        ]
        state = _base_state(hypotheses=hyps, convergence_threshold=0.9)
        result = update_convergence(state)
        # Equal confidences → low convergence
        assert result["convergence_score"] < 0.5

    def test_dominant_hypothesis_high_convergence(self) -> None:
        hyps = [
            {"hypothesis_id": "h1", "description": "Strong hyp", "confidence": 0.9, "status": "active",
             "supporting_evidence_ids": [], "contradicting_evidence_ids": []},
            {"hypothesis_id": "h2", "description": "Weak hyp", "confidence": 0.1, "status": "active",
             "supporting_evidence_ids": [], "contradicting_evidence_ids": []},
        ]
        state = _base_state(hypotheses=hyps, convergence_threshold=0.8)
        result = update_convergence(state)
        assert result["convergence_score"] >= 0.8

    def test_increments_iteration_count(self) -> None:
        state = _base_state(iteration_count=2, hypotheses=[])
        result = update_convergence(state)
        assert result["iteration_count"] == 3


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

    def test_graph_compiles(self) -> None:
        graph = build_reasoning_graph(lambda: self._mock_factory())
        assert graph is not None

    def test_graph_runs_to_completion(self) -> None:
        mock_llm = self._mock_factory()
        graph = build_reasoning_graph(lambda: mock_llm)
        initial: ReasoningState = _base_state()
        result = graph.invoke(initial)
        assert result["iteration_count"] >= 1
        assert len(result["hypotheses"]) > 0
        assert result["convergence_score"] >= 0.0

    def test_max_iteration_safety(self) -> None:
        """Graph always terminates within max_iterations."""
        # Make convergence impossible by returning equal-confidence hypotheses
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

    def test_hypotheses_have_required_fields(self) -> None:
        mock_llm = self._mock_factory()
        graph = build_reasoning_graph(lambda: mock_llm)
        result = graph.invoke(_base_state())
        for h in result["hypotheses"]:
            assert "hypothesis_id" in h
            assert "description" in h
            assert "confidence" in h
            assert "status" in h


# ── Runner Tests ─────────────────────────────────────────────────────────────


class TestRunner:
    def test_run_reasoning_returns_valid_state(self) -> None:
        mock_llm = _mock_llm_response(_sample_hypotheses_json())
        # Create a minimal situation
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
        )

        assert result["situation_id"] == str(sit.situation_id)
        assert result["iteration_count"] >= 1
        assert isinstance(result["hypotheses"], list)
        assert 0.0 <= result["convergence_score"] <= 1.0
