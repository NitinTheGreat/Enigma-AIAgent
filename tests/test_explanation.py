"""Tests for Phase 6 + 6.1: Explainability & XAI Reinforcement.

Tests cover:
- ExplanationSnapshot and ExplanationSection model correctness
- UNKNOWN-dominant explanations (undecided path)
- Converged explanations (decided path)
- "What would change my mind" correctness
- No hallucinated fields (every bullet traceable to inputs)
- Determinism (same input → same explanation)
- ExplanationFormatter plain-text output
- ExplanationFormatter LLM fallback

Phase 6.1:
- Contribution score correctness and bounds
- Counterfactual correctness under known thresholds
- Temporal summaries reflecting belief dynamics
- Integrity validator catching invalid claims
- Role-based filtering correctness
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from enigma_reason.domain.explanation import (
    ContributionDirection,
    Counterfactual,
    ExplanationRole,
    ExplanationSection,
    ExplanationSnapshot,
    SectionType,
    TemporalEvolution,
    filter_explanation_for_role,
)
from enigma_reason.domain.hypothesis import (
    UNKNOWN_HYPOTHESIS_ID,
    make_unknown_hypothesis,
)
from enigma_reason.domain.reasoning import SituationReasoningSnapshot, Trend
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.explain.builder import (
    ExplanationIntegrityError,
    build_explanation,
    validate_explanation_integrity,
)
from enigma_reason.explain.formatter import ExplanationFormatter


# ── Helpers ──────────────────────────────────────────────────────────────────


def _temporal(
    event_count: int = 5,
    duration: float = 120.0,
    rate: float = 2.5,
) -> SituationTemporalSnapshot:
    return SituationTemporalSnapshot(
        situation_id="test-sit-001",
        event_count=event_count,
        active_duration_seconds=duration,
        event_rate_per_minute=rate,
        last_event_age_seconds=10.0,
        mean_interval_seconds=30.0,
        burst_detected=False,
        quiet_detected=False,
    )


def _reasoning(
    evidence_count: int = 5,
    anomaly: float = 0.5,
    diversity: int = 2,
    trend: str = "stable",
    burst: bool = False,
    quiet: bool = False,
    confidence: float = 0.5,
) -> SituationReasoningSnapshot:
    return SituationReasoningSnapshot(
        situation_id="test-sit-001",
        evidence_count=evidence_count,
        event_rate=2.5,
        burst_detected=burst,
        quiet_detected=quiet,
        confidence_level=confidence,
        trend=Trend(trend),
        source_diversity=diversity,
        mean_anomaly_score=anomaly,
    )


def _hyp(desc: str, conf: float, hid: str | None = None, status: str = "active", **kw) -> dict:
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


def _state(hypotheses: list[dict], **overrides) -> dict[str, Any]:
    s: dict[str, Any] = {
        "situation_id": "test-sit-001",
        "hypotheses": hypotheses,
        "convergence_score": overrides.pop("convergence_score", 0.3),
        "belief_stability_score": overrides.pop("belief_stability_score", 0.5),
        "iteration_count": overrides.pop("iteration_count", 2),
        "undecided_iterations": overrides.pop("undecided_iterations", 0),
        "last_confidence_shift": overrides.pop("last_confidence_shift", 0.05),
    }
    s.update(overrides)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 tests (preserved from original)
# ══════════════════════════════════════════════════════════════════════════════


class TestExplanationModels:
    def test_section_type_enum_values(self) -> None:
        assert SectionType.SUMMARY == "SUMMARY"
        assert SectionType.WHY_UNKNOWN == "WHY_UNKNOWN"
        assert SectionType.WHAT_WOULD_CHANGE_MY_MIND == "WHAT_WOULD_CHANGE_MY_MIND"

    def test_section_creation(self) -> None:
        s = ExplanationSection(
            section_type=SectionType.SUMMARY,
            title="Current Belief State",
            bullet_points=["Converged on hypothesis X"],
            referenced_fields=["convergence_score"],
        )
        assert s.section_type == SectionType.SUMMARY
        assert len(s.bullet_points) == 1

    def test_snapshot_is_frozen(self) -> None:
        snap = ExplanationSnapshot(
            situation_id="test",
            dominant_confidence=0.5,
            convergence_score=0.3,
        )
        with pytest.raises(Exception):
            snap.convergence_score = 0.9  # type: ignore

    def test_snapshot_defaults(self) -> None:
        snap = ExplanationSnapshot(situation_id="test")
        assert snap.undecided is True
        assert snap.dominant_hypothesis_id is None
        assert snap.explanation_sections == []


class TestUndecidedExplanation:
    def test_unknown_dominant_is_undecided(self) -> None:
        hyps = [
            make_unknown_hypothesis(0.6),
            _hyp("Some threat hypothesis present", 0.3),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        assert exp.undecided is True
        assert exp.dominant_hypothesis_id == UNKNOWN_HYPOTHESIS_ID

    def test_undecided_has_why_unknown_section(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(evidence_count=1, diversity=1),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.WHY_UNKNOWN in types

    def test_sparse_evidence_mentioned(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(evidence_count=2),
            _temporal(),
        )
        why = next(s for s in exp.explanation_sections if s.section_type == SectionType.WHY_UNKNOWN)
        assert any("sparse" in bp.lower() for bp in why.bullet_points)

    def test_low_diversity_mentioned(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(diversity=1),
            _temporal(),
        )
        why = next(s for s in exp.explanation_sections if s.section_type == SectionType.WHY_UNKNOWN)
        assert any("diversity" in bp.lower() for bp in why.bullet_points)

    def test_no_converged_sections_when_undecided(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.SUPPORTING_EVIDENCE not in types
        assert SectionType.CONTRADICTING_EVIDENCE not in types


class TestConvergedExplanation:
    def test_converged_is_not_undecided(self) -> None:
        hyps = [
            _hyp("Confirmed threat pattern match", 0.9, status="converged", dominant_iterations=3),
            make_unknown_hypothesis(0.1),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.9),
            _reasoning(anomaly=0.8, burst=True, diversity=3, trend="escalating"),
            _temporal(),
        )
        assert exp.undecided is False
        assert exp.dominant_confidence == 0.9

    def test_converged_has_supporting_evidence(self) -> None:
        hyps = [
            _hyp("Confirmed sustained activity", 0.85, status="converged", dominant_iterations=3),
            make_unknown_hypothesis(0.1),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.85),
            _reasoning(anomaly=0.8, burst=True, diversity=3, trend="escalating"),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.SUPPORTING_EVIDENCE in types
        assert SectionType.CONTRADICTING_EVIDENCE in types

    def test_supporting_mentions_burst(self) -> None:
        hyps = [
            _hyp("Burst-related activity pattern", 0.85, status="converged", dominant_iterations=3),
            make_unknown_hypothesis(0.1),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.85),
            _reasoning(burst=True, anomaly=0.8),
            _temporal(),
        )
        sup = next(s for s in exp.explanation_sections if s.section_type == SectionType.SUPPORTING_EVIDENCE)
        assert any("burst" in bp.lower() for bp in sup.bullet_points)

    def test_no_why_unknown_when_converged(self) -> None:
        hyps = [
            _hyp("Confirmed threat pattern match", 0.9, status="converged", dominant_iterations=3),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.9),
            _reasoning(),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.WHY_UNKNOWN not in types


class TestWhatWouldChange:
    def test_undecided_mentions_more_evidence(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(evidence_count=2),
            _temporal(),
        )
        change = next(
            s for s in exp.explanation_sections
            if s.section_type == SectionType.WHAT_WOULD_CHANGE_MY_MIND
        )
        assert any("evidence" in bp.lower() for bp in change.bullet_points)

    def test_undecided_mentions_diversity(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(diversity=1),
            _temporal(),
        )
        change = next(
            s for s in exp.explanation_sections
            if s.section_type == SectionType.WHAT_WOULD_CHANGE_MY_MIND
        )
        assert any("diversity" in bp.lower() for bp in change.bullet_points)

    def test_undecided_mentions_sustained_dominance(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        change = next(
            s for s in exp.explanation_sections
            if s.section_type == SectionType.WHAT_WOULD_CHANGE_MY_MIND
        )
        assert any("sustained" in bp.lower() or "dominance" in bp.lower() for bp in change.bullet_points)

    def test_converged_mentions_unknown_threat(self) -> None:
        hyps = [
            _hyp("Confirmed pattern match result", 0.9, status="converged"),
            make_unknown_hypothesis(0.15),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.9),
            _reasoning(),
            _temporal(),
        )
        change = next(
            s for s in exp.explanation_sections
            if s.section_type == SectionType.WHAT_WOULD_CHANGE_MY_MIND
        )
        assert any("unknown" in bp.lower() for bp in change.bullet_points)


class TestDeterminismAndIntegrity:
    def test_same_input_same_output(self) -> None:
        hyps = [
            make_unknown_hypothesis(0.5),
            _hyp("Correlated anomaly cluster signal", 0.4),
        ]
        state = _state(hyps, convergence_score=0.3)
        rs = _reasoning()
        ts = _temporal()

        exp1 = build_explanation(state, rs, ts)
        exp2 = build_explanation(state, rs, ts)

        assert exp1.undecided == exp2.undecided
        assert exp1.dominant_confidence == exp2.dominant_confidence
        assert exp1.convergence_score == exp2.convergence_score
        assert len(exp1.explanation_sections) == len(exp2.explanation_sections)
        for s1, s2 in zip(exp1.explanation_sections, exp2.explanation_sections):
            assert s1.section_type == s2.section_type
            assert s1.bullet_points == s2.bullet_points

    def test_no_hallucinated_section_types(self) -> None:
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        valid_types = set(SectionType)
        for s in exp.explanation_sections:
            assert s.section_type in valid_types

    def test_every_section_has_referenced_fields(self) -> None:
        hyps = [
            _hyp("Confirmed threat pattern match", 0.9, status="converged", dominant_iterations=3),
            make_unknown_hypothesis(0.1),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.9),
            _reasoning(burst=True, anomaly=0.8),
            _temporal(),
        )
        for s in exp.explanation_sections:
            assert len(s.referenced_fields) > 0, f"Section {s.section_type} has no referenced_fields"

    def test_always_has_summary_and_change(self) -> None:
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.SUMMARY in types
        assert SectionType.WHAT_WOULD_CHANGE_MY_MIND in types

    def test_confidence_rationale_always_present(self) -> None:
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.CONFIDENCE_RATIONALE in types

    def test_explanation_snapshot_immutable(self) -> None:
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        with pytest.raises(Exception):
            exp.undecided = False  # type: ignore


class TestExplanationFormatter:
    def test_plain_format_includes_status(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        text = ExplanationFormatter.format_plain(exp)
        assert "UNDECIDED" in text

    def test_plain_format_includes_sections(self) -> None:
        hyps = [
            _hyp("Confirmed threat pattern match", 0.9, status="converged"),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.9),
            _reasoning(burst=True),
            _temporal(),
        )
        text = ExplanationFormatter.format_plain(exp)
        assert "Supporting Factors" in text
        assert "Contradicting Factors" in text
        assert "CONVERGED" in text

    def test_llm_formatter_fallback_on_error(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API down")
        formatter = ExplanationFormatter(lambda: mock_llm)

        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        text = formatter.format(exp)
        assert "UNDECIDED" in text

    def test_llm_formatter_uses_llm_when_available(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = SimpleNamespace(
            content="The system has not yet reached a conclusion."
        )
        formatter = ExplanationFormatter(lambda: mock_llm)

        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        text = formatter.format(exp)
        assert "not yet reached" in text
        mock_llm.invoke.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6.1 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestContributionScoring:
    """Tests that contribution_score and contribution_direction are correct."""

    def test_supporting_has_contribution_score(self) -> None:
        hyps = [
            _hyp("Confirmed active threat", 0.85, status="converged", dominant_iterations=3),
            make_unknown_hypothesis(0.1),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.85),
            _reasoning(burst=True, anomaly=0.8, diversity=3, trend="escalating"),
            _temporal(),
        )
        sup = next(s for s in exp.explanation_sections if s.section_type == SectionType.SUPPORTING_EVIDENCE)
        assert sup.contribution_score is not None
        assert 0.0 <= sup.contribution_score <= 1.0
        assert sup.contribution_direction == ContributionDirection.SUPPORTING

    def test_contradicting_has_contribution_score(self) -> None:
        hyps = [
            _hyp("Confirmed active threat", 0.85, status="converged", dominant_iterations=3),
            make_unknown_hypothesis(0.1),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.85),
            _reasoning(quiet=True, anomaly=0.2, diversity=1, trend="deescalating"),
            _temporal(),
        )
        con = next(s for s in exp.explanation_sections if s.section_type == SectionType.CONTRADICTING_EVIDENCE)
        assert con.contribution_score is not None
        assert 0.0 <= con.contribution_score <= 1.0
        assert con.contribution_direction == ContributionDirection.OPPOSING

    def test_contribution_score_bounds(self) -> None:
        """Contribution scores must always be in [0, 1]."""
        hyps = [
            _hyp("Max-boosted hypothesis threat", 0.9, status="converged", dominant_iterations=5),
            make_unknown_hypothesis(0.05),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.95),
            _reasoning(burst=True, anomaly=0.95, diversity=5, trend="escalating", confidence=0.9),
            _temporal(),
        )
        for s in exp.explanation_sections:
            if s.contribution_score is not None:
                assert 0.0 <= s.contribution_score <= 1.0

    def test_higher_anomaly_yields_higher_supporting_score(self) -> None:
        """Higher anomaly + burst should produce higher supporting contribution."""
        hyps_low = [
            _hyp("Threat hypothesis low anomaly", 0.7, status="converged", dominant_iterations=3),
        ]
        exp_low = build_explanation(
            _state(hyps_low, convergence_score=0.7),
            _reasoning(anomaly=0.3, burst=False, diversity=2),
            _temporal(),
        )
        hyps_high = [
            _hyp("Threat hypothesis high anomaly", 0.9, status="converged", dominant_iterations=3),
        ]
        exp_high = build_explanation(
            _state(hyps_high, convergence_score=0.9),
            _reasoning(anomaly=0.9, burst=True, diversity=4, trend="escalating"),
            _temporal(),
        )
        sup_low = next(s for s in exp_low.explanation_sections if s.section_type == SectionType.SUPPORTING_EVIDENCE)
        sup_high = next(s for s in exp_high.explanation_sections if s.section_type == SectionType.SUPPORTING_EVIDENCE)
        assert sup_high.contribution_score > sup_low.contribution_score

    def test_why_unknown_direction_is_neutral(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        why = next(s for s in exp.explanation_sections if s.section_type == SectionType.WHY_UNKNOWN)
        assert why.contribution_direction == ContributionDirection.NEUTRAL


class TestCounterfactuals:
    """Tests for structured counterfactual projections."""

    def test_counterfactuals_section_present(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(evidence_count=2, diversity=1),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.COUNTERFACTUALS in types

    def test_counterfactual_for_sparse_evidence(self) -> None:
        """Sparse evidence should generate 'more signals' counterfactual."""
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(evidence_count=2),
            _temporal(),
        )
        cf_section = next(s for s in exp.explanation_sections if s.section_type == SectionType.COUNTERFACTUALS)
        assert len(cf_section.counterfactuals) >= 1
        # Should mention evidence
        assert any("signal" in cf.missing_condition.lower() for cf in cf_section.counterfactuals)

    def test_counterfactual_for_low_diversity(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(diversity=1),
            _temporal(),
        )
        cf_section = next(s for s in exp.explanation_sections if s.section_type == SectionType.COUNTERFACTUALS)
        assert any("source" in cf.missing_condition.lower() for cf in cf_section.counterfactuals)

    def test_counterfactual_for_no_burst(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(burst=False),
            _temporal(),
        )
        cf_section = next(s for s in exp.explanation_sections if s.section_type == SectionType.COUNTERFACTUALS)
        assert any("burst" in cf.missing_condition.lower() for cf in cf_section.counterfactuals)

    def test_counterfactual_delta_bounds(self) -> None:
        """All confidence_delta_estimate values must be in [-1, 1]."""
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(evidence_count=1, diversity=1),
            _temporal(),
        )
        cf_section = next(s for s in exp.explanation_sections if s.section_type == SectionType.COUNTERFACTUALS)
        for cf in cf_section.counterfactuals:
            assert -1.0 <= cf.confidence_delta_estimate <= 1.0

    def test_counterfactual_sustained_dominance(self) -> None:
        """Hypothesis with low dominant_iterations should get dominance counterfactual."""
        hyps = [
            _hyp("Almost dominant hypothesis test", 0.7, dominant_iterations=1),
            make_unknown_hypothesis(0.3),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.5),
            _reasoning(),
            _temporal(),
        )
        cf_section = next(s for s in exp.explanation_sections if s.section_type == SectionType.COUNTERFACTUALS)
        assert any("dominance" in cf.missing_condition.lower() or "dominant" in cf.missing_condition.lower()
                    for cf in cf_section.counterfactuals)

    def test_counterfactual_quiet_period_when_converged(self) -> None:
        """Converged + no quiet → quiet period counterfactual."""
        hyps = [
            _hyp("Active threat confirmed now", 0.9, status="converged", dominant_iterations=3),
        ]
        exp = build_explanation(
            _state(hyps, convergence_score=0.9),
            _reasoning(quiet=False),
            _temporal(),
        )
        cf_section = next(s for s in exp.explanation_sections if s.section_type == SectionType.COUNTERFACTUALS)
        assert any("quiet" in cf.missing_condition.lower() for cf in cf_section.counterfactuals)

    def test_counterfactual_determinism(self) -> None:
        """Same inputs must produce identical counterfactuals."""
        hyps = [make_unknown_hypothesis(0.5)]
        state = _state(hyps, convergence_score=0.0)
        rs = _reasoning(evidence_count=2, diversity=1)
        ts = _temporal()

        exp1 = build_explanation(state, rs, ts)
        exp2 = build_explanation(state, rs, ts)

        cf1 = next(s for s in exp1.explanation_sections if s.section_type == SectionType.COUNTERFACTUALS)
        cf2 = next(s for s in exp2.explanation_sections if s.section_type == SectionType.COUNTERFACTUALS)
        assert len(cf1.counterfactuals) == len(cf2.counterfactuals)
        for c1, c2 in zip(cf1.counterfactuals, cf2.counterfactuals):
            assert c1.missing_condition == c2.missing_condition
            assert c1.confidence_delta_estimate == c2.confidence_delta_estimate


class TestTemporalEvolution:
    """Tests for temporal belief evolution summaries."""

    def test_temporal_evolution_present(self) -> None:
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution is not None

    def test_temporal_section_present(self) -> None:
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.TEMPORAL_EVOLUTION in types

    def test_rising_trend_with_positive_velocity(self) -> None:
        hyps = [_hyp("Rising threat signal pattern", 0.7, belief_velocity=0.05)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution.confidence_trend == "rising"

    def test_falling_trend_with_negative_velocity(self) -> None:
        hyps = [_hyp("Falling threat signal pattern", 0.3, belief_velocity=-0.05)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution.confidence_trend == "falling"

    def test_flat_trend_with_zero_velocity(self) -> None:
        hyps = [_hyp("Stable threat signal pattern", 0.5, belief_velocity=0.0)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution.confidence_trend == "flat"

    def test_velocity_summary_slow(self) -> None:
        hyps = [_hyp("Slow-moving hypothesis test", 0.5, belief_velocity=0.01)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution.belief_velocity_summary == "slow"

    def test_velocity_summary_fast(self) -> None:
        hyps = [_hyp("Fast-moving hypothesis test", 0.5, belief_velocity=0.15)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution.belief_velocity_summary == "fast"

    def test_undecided_duration_tracked(self) -> None:
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, undecided_iterations=4),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution.undecided_duration == 4

    def test_stability_label_stable(self) -> None:
        hyps = [_hyp("Stable hypothesis test data", 0.5)]
        exp = build_explanation(
            _state(hyps, belief_stability_score=0.8),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution.stability_label == "stable"

    def test_stability_label_volatile(self) -> None:
        hyps = [_hyp("Volatile hypothesis test data", 0.5)]
        exp = build_explanation(
            _state(hyps, belief_stability_score=0.2),
            _reasoning(),
            _temporal(),
        )
        assert exp.temporal_evolution.stability_label == "volatile"


class TestIntegrityValidator:
    """Tests for the fail-closed integrity validator."""

    def test_valid_explanation_passes(self) -> None:
        """Normal explanation should pass validation without errors."""
        hyps = [make_unknown_hypothesis(0.5)]
        # This should not raise
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        assert exp is not None

    def test_unknown_field_raises_error(self) -> None:
        """Referencing an unknown field must raise ExplanationIntegrityError."""
        bad_snapshot = ExplanationSnapshot(
            situation_id="test",
            explanation_sections=[
                ExplanationSection(
                    section_type=SectionType.SUMMARY,
                    title="Bad Section",
                    bullet_points=["Some claim"],
                    referenced_fields=["nonexistent_hallucinated_field"],
                ),
            ],
        )
        with pytest.raises(ExplanationIntegrityError) as exc_info:
            validate_explanation_integrity(bad_snapshot)
        assert "nonexistent_hallucinated_field" in str(exc_info.value)

    def test_empty_bullets_raises_error(self) -> None:
        """Section with no bullet points must raise ExplanationIntegrityError."""
        bad_snapshot = ExplanationSnapshot(
            situation_id="test",
            explanation_sections=[
                ExplanationSection(
                    section_type=SectionType.SUMMARY,
                    title="Empty Section",
                    bullet_points=[],
                    referenced_fields=["convergence_score"],
                ),
            ],
        )
        with pytest.raises(ExplanationIntegrityError) as exc_info:
            validate_explanation_integrity(bad_snapshot)
        assert "no bullet points" in str(exc_info.value).lower()

    def test_integrity_error_contains_violations(self) -> None:
        """ExplanationIntegrityError must carry a list of violations."""
        bad_snapshot = ExplanationSnapshot(
            situation_id="test",
            explanation_sections=[
                ExplanationSection(
                    section_type=SectionType.SUMMARY,
                    title="Bad Section",
                    bullet_points=["claim"],
                    referenced_fields=["fake_field_a", "fake_field_b"],
                ),
            ],
        )
        with pytest.raises(ExplanationIntegrityError) as exc_info:
            validate_explanation_integrity(bad_snapshot)
        assert len(exc_info.value.violations) == 2

    def test_valid_fields_pass(self) -> None:
        """All known fields should pass validation."""
        good_snapshot = ExplanationSnapshot(
            situation_id="test",
            explanation_sections=[
                ExplanationSection(
                    section_type=SectionType.SUMMARY,
                    title="Good Section",
                    bullet_points=["Valid claim"],
                    referenced_fields=["convergence_score", "evidence_count", "trend"],
                ),
            ],
        )
        # Should not raise
        validate_explanation_integrity(good_snapshot)


class TestRoleBasedViews:
    """Tests for role-based explanation filtering."""

    def _full_explanation(self) -> ExplanationSnapshot:
        hyps = [
            _hyp("Confirmed threat pattern match", 0.9, status="converged", dominant_iterations=3),
            make_unknown_hypothesis(0.1),
        ]
        return build_explanation(
            _state(hyps, convergence_score=0.9),
            _reasoning(burst=True, anomaly=0.8, diversity=3),
            _temporal(),
        )

    def test_analyst_sees_all_sections(self) -> None:
        exp = self._full_explanation()
        filtered = filter_explanation_for_role(exp, ExplanationRole.ANALYST)
        assert len(filtered.explanation_sections) == len(exp.explanation_sections)

    def test_manager_sees_subset(self) -> None:
        exp = self._full_explanation()
        filtered = filter_explanation_for_role(exp, ExplanationRole.MANAGER)
        types = {s.section_type for s in filtered.explanation_sections}
        assert SectionType.SUMMARY in types
        assert SectionType.CONFIDENCE_RATIONALE in types
        assert SectionType.COUNTERFACTUALS in types
        assert SectionType.TEMPORAL_EVOLUTION in types
        # Manager should NOT see raw supporting/contradicting evidence
        assert SectionType.SUPPORTING_EVIDENCE not in types
        assert SectionType.CONTRADICTING_EVIDENCE not in types

    def test_auditor_sees_deterministic_only(self) -> None:
        exp = self._full_explanation()
        filtered = filter_explanation_for_role(exp, ExplanationRole.AUDITOR)
        types = {s.section_type for s in filtered.explanation_sections}
        assert SectionType.SUMMARY in types
        assert SectionType.CONFIDENCE_RATIONALE in types
        assert SectionType.WHAT_WOULD_CHANGE_MY_MIND in types
        # Auditor should NOT see counterfactuals or temporal evolution
        assert SectionType.COUNTERFACTUALS not in types
        assert SectionType.TEMPORAL_EVOLUTION not in types

    def test_auditor_no_temporal_evolution(self) -> None:
        """Auditor view should strip temporal_evolution."""
        exp = self._full_explanation()
        filtered = filter_explanation_for_role(exp, ExplanationRole.AUDITOR)
        assert filtered.temporal_evolution is None

    def test_filtered_preserves_core_fields(self) -> None:
        """Filtering must not change core snapshot fields."""
        exp = self._full_explanation()
        for role in ExplanationRole:
            filtered = filter_explanation_for_role(exp, role)
            assert filtered.situation_id == exp.situation_id
            assert filtered.dominant_confidence == exp.dominant_confidence
            assert filtered.convergence_score == exp.convergence_score
            assert filtered.undecided == exp.undecided

    def test_filtered_snapshot_is_frozen(self) -> None:
        exp = self._full_explanation()
        filtered = filter_explanation_for_role(exp, ExplanationRole.MANAGER)
        with pytest.raises(Exception):
            filtered.undecided = True  # type: ignore

    def test_manager_keeps_temporal_evolution(self) -> None:
        exp = self._full_explanation()
        filtered = filter_explanation_for_role(exp, ExplanationRole.MANAGER)
        assert filtered.temporal_evolution is not None
