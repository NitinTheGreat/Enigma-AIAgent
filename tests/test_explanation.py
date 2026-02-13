"""Tests for Phase 6: Explainability & Human Trust Layer.

Tests cover:
- ExplanationSnapshot and ExplanationSection model correctness
- UNKNOWN-dominant explanations (undecided path)
- Converged explanations (decided path)
- "What would change my mind" correctness
- No hallucinated fields (every bullet traceable to inputs)
- Determinism (same input → same explanation)
- ExplanationFormatter plain-text output
- ExplanationFormatter LLM fallback
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from enigma_reason.domain.explanation import (
    ExplanationSection,
    ExplanationSnapshot,
    SectionType,
)
from enigma_reason.domain.hypothesis import (
    UNKNOWN_HYPOTHESIS_ID,
    make_unknown_hypothesis,
)
from enigma_reason.domain.reasoning import SituationReasoningSnapshot, Trend
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.explain.builder import build_explanation
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


# ── Model Tests ──────────────────────────────────────────────────────────────


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


# ── UNKNOWN-Dominant Explanation Tests ───────────────────────────────────────


class TestUndecidedExplanation:
    def test_unknown_dominant_is_undecided(self) -> None:
        """When UNKNOWN is highest confidence, explanation is undecided."""
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
        """Undecided explanation must include WHY_UNKNOWN section."""
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(evidence_count=1, diversity=1),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.WHY_UNKNOWN in types

    def test_sparse_evidence_mentioned(self) -> None:
        """When evidence < 3, WHY_UNKNOWN must mention sparse evidence."""
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(evidence_count=2),
            _temporal(),
        )
        why = next(s for s in exp.explanation_sections if s.section_type == SectionType.WHY_UNKNOWN)
        assert any("sparse" in bp.lower() for bp in why.bullet_points)

    def test_low_diversity_mentioned(self) -> None:
        """When diversity <= 1, WHY_UNKNOWN must mention it."""
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(diversity=1),
            _temporal(),
        )
        why = next(s for s in exp.explanation_sections if s.section_type == SectionType.WHY_UNKNOWN)
        assert any("diversity" in bp.lower() for bp in why.bullet_points)

    def test_no_converged_sections_when_undecided(self) -> None:
        """Undecided explanations must NOT have SUPPORTING_EVIDENCE."""
        hyps = [make_unknown_hypothesis(0.6)]
        exp = build_explanation(
            _state(hyps, convergence_score=0.0),
            _reasoning(),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.SUPPORTING_EVIDENCE not in types
        assert SectionType.CONTRADICTING_EVIDENCE not in types


# ── Converged Explanation Tests ──────────────────────────────────────────────


class TestConvergedExplanation:
    def test_converged_is_not_undecided(self) -> None:
        """Converged hypothesis → undecided=False."""
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
        """Converged explanations must NOT have WHY_UNKNOWN."""
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


# ── What Would Change My Mind Tests ──────────────────────────────────────────


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
        """Converged explanation should mention UNKNOWN could revert assessment."""
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


# ── Determinism & No-Hallucination Tests ─────────────────────────────────────


class TestDeterminismAndIntegrity:
    def test_same_input_same_output(self) -> None:
        """Determinism: identical inputs must produce identical explanations."""
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
        """All section types must be from the SectionType enum."""
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
        """Every section must declare which reasoning fields it references."""
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
        """Every explanation must have SUMMARY and WHAT_WOULD_CHANGE_MY_MIND."""
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
        """CONFIDENCE_RATIONALE must always be included."""
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        types = [s.section_type for s in exp.explanation_sections]
        assert SectionType.CONFIDENCE_RATIONALE in types

    def test_explanation_snapshot_immutable(self) -> None:
        """ExplanationSnapshot must be frozen — no mutation after creation."""
        hyps = [make_unknown_hypothesis(0.5)]
        exp = build_explanation(
            _state(hyps),
            _reasoning(),
            _temporal(),
        )
        with pytest.raises(Exception):
            exp.undecided = False  # type: ignore


# ── Formatter Tests ──────────────────────────────────────────────────────────


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
        """LLM failure must fall back to plain text."""
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
        assert "UNDECIDED" in text  # fell back to plain

    def test_llm_formatter_uses_llm_when_available(self) -> None:
        """When LLM works, its output is used."""
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
