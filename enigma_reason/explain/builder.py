"""ExplanationBuilder — pure, deterministic explanation generator.

Accepts completed ReasoningState + Phase 2/4 snapshots and produces
an ExplanationSnapshot describing what the system believes and why.

Rules:
    - NEVER modifies reasoning state
    - NEVER uses an LLM
    - NEVER invents facts not present in the inputs
    - Every bullet point is traceable to a specific field
    - Output is deterministic: same input → same explanation
"""

from __future__ import annotations

import logging
from typing import Any

from enigma_reason.domain.explanation import (
    ExplanationSection,
    ExplanationSnapshot,
    SectionType,
)
from enigma_reason.domain.hypothesis import UNKNOWN_HYPOTHESIS_ID
from enigma_reason.domain.reasoning import SituationReasoningSnapshot
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.graph.state import ReasoningState

logger = logging.getLogger(__name__)


def build_explanation(
    reasoning_state: dict[str, Any],
    reasoning_snapshot: SituationReasoningSnapshot,
    temporal_snapshot: SituationTemporalSnapshot,
) -> ExplanationSnapshot:
    """Build a structured explanation from completed reasoning state.

    This is the public interface for Phase 6.  It reads the final
    ReasoningState (as a dict) plus the Phase 2 and Phase 4 snapshots,
    and produces an immutable ExplanationSnapshot.

    Args:
        reasoning_state: Final ReasoningState dict from the graph runner.
        reasoning_snapshot: Phase 4 SituationReasoningSnapshot.
        temporal_snapshot: Phase 2 SituationTemporalSnapshot.

    Returns:
        Frozen ExplanationSnapshot with ordered explanation sections.
    """
    hypotheses = reasoning_state.get("hypotheses", [])
    convergence = reasoning_state.get("convergence_score", 0.0)
    stability = reasoning_state.get("belief_stability_score", 0.0)
    iteration_count = reasoning_state.get("iteration_count", 0)
    situation_id = reasoning_state.get("situation_id", "")

    # ── Identify dominant hypothesis ────────────────────────────────────
    active = [h for h in hypotheses if h.get("status") in ("active", "converged")]
    converged = [h for h in hypotheses if h.get("status") == "converged"]

    if converged:
        dominant = converged[0]
    elif active:
        dominant = max(active, key=lambda h: h.get("confidence", 0.0))
    else:
        dominant = None

    dominant_id = dominant.get("hypothesis_id") if dominant else None
    dominant_desc = dominant.get("description") if dominant else None
    dominant_conf = dominant.get("confidence", 0.0) if dominant else 0.0
    is_undecided = (
        dominant_id is None
        or dominant_id == UNKNOWN_HYPOTHESIS_ID
        or not converged
    )

    # ── Build sections ──────────────────────────────────────────────────
    sections: list[ExplanationSection] = []

    sections.append(_build_summary(
        dominant, is_undecided, convergence, iteration_count,
    ))

    if is_undecided:
        sections.append(_build_why_unknown(
            hypotheses, reasoning_snapshot, temporal_snapshot,
            convergence, stability,
        ))
    else:
        sections.append(_build_supporting_evidence(
            dominant, reasoning_snapshot, temporal_snapshot,
        ))
        sections.append(_build_contradicting_evidence(
            dominant, hypotheses, reasoning_snapshot, temporal_snapshot,
        ))

    sections.append(_build_confidence_rationale(
        dominant, hypotheses, reasoning_snapshot, convergence, stability,
    ))

    sections.append(_build_what_would_change(
        is_undecided, dominant, hypotheses, reasoning_snapshot, temporal_snapshot,
    ))

    return ExplanationSnapshot(
        situation_id=situation_id,
        dominant_hypothesis_id=dominant_id,
        dominant_hypothesis_description=dominant_desc,
        dominant_confidence=round(dominant_conf, 4),
        convergence_score=round(convergence, 4),
        belief_stability_score=round(stability, 4),
        undecided=is_undecided,
        iteration_count=iteration_count,
        explanation_sections=sections,
    )


# ── Section Builders ─────────────────────────────────────────────────────────


def _build_summary(
    dominant: dict | None,
    undecided: bool,
    convergence: float,
    iteration_count: int,
) -> ExplanationSection:
    """SUMMARY: What the system currently believes."""
    bullets = []

    if undecided:
        if dominant and dominant.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID:
            bullets.append("System is explicitly undecided: UNKNOWN hypothesis is dominant")
        else:
            bullets.append("System has not reached convergence on any hypothesis")
        bullets.append(f"Convergence score: {convergence:.2f} (below threshold)")
    else:
        desc = dominant.get("description", "N/A") if dominant else "N/A"
        conf = dominant.get("confidence", 0.0) if dominant else 0.0
        bullets.append(f"Converged on: {desc}")
        bullets.append(f"Confidence: {conf:.2f}, convergence: {convergence:.2f}")

    bullets.append(f"Completed {iteration_count} reasoning iteration(s)")

    return ExplanationSection(
        section_type=SectionType.SUMMARY,
        title="Current Belief State",
        bullet_points=bullets,
        referenced_fields=["convergence_score", "iteration_count", "dominant_hypothesis"],
    )


def _build_why_unknown(
    hypotheses: list[dict],
    rs: SituationReasoningSnapshot,
    ts: SituationTemporalSnapshot,
    convergence: float,
    stability: float,
) -> ExplanationSection:
    """WHY_UNKNOWN: Why the system remains undecided."""
    bullets = []

    # Sparse evidence
    if rs.evidence_count < 3:
        bullets.append(f"Evidence is sparse: only {rs.evidence_count} signal(s) observed")

    # Low diversity
    if rs.source_diversity <= 1:
        bullets.append(f"Source diversity is low: {rs.source_diversity} distinct source(s)")

    # Flat confidence
    active = [h for h in hypotheses if h.get("status") == "active"
              and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID]
    if len(active) >= 2:
        confs = [h["confidence"] for h in active]
        spread = max(confs) - min(confs)
        if spread < 0.15:
            bullets.append(f"Hypothesis confidence spread is flat: {spread:.2f}")

    # Low stability
    if stability < 0.5:
        bullets.append(f"Belief stability is low: {stability:.2f} (confidence still shifting)")

    # Low convergence
    if convergence < 0.3:
        bullets.append(f"Convergence is very low: {convergence:.2f}")

    if not bullets:
        bullets.append("No single hypothesis has sustained dominance over UNKNOWN")

    return ExplanationSection(
        section_type=SectionType.WHY_UNKNOWN,
        title="Why the System is Undecided",
        bullet_points=bullets,
        referenced_fields=[
            "evidence_count", "source_diversity", "confidence_spread",
            "belief_stability_score", "convergence_score",
        ],
    )


def _build_supporting_evidence(
    dominant: dict,
    rs: SituationReasoningSnapshot,
    ts: SituationTemporalSnapshot,
) -> ExplanationSection:
    """SUPPORTING_EVIDENCE: Factors reinforcing the dominant hypothesis."""
    bullets = []

    if rs.burst_detected:
        bullets.append("Burst activity detected: supports escalation-related hypotheses")

    if rs.mean_anomaly_score > 0.6:
        bullets.append(f"Mean anomaly score is elevated: {rs.mean_anomaly_score:.2f}")

    if rs.trend.value == "escalating":
        bullets.append("Trend is escalating: consistent with sustained activity")

    if rs.evidence_count >= 5:
        bullets.append(f"Evidence volume is sufficient: {rs.evidence_count} signal(s)")

    if rs.source_diversity >= 3:
        bullets.append(f"Source diversity is high: {rs.source_diversity} distinct sources")

    if rs.confidence_level > 0.6:
        bullets.append(f"Deterministic confidence level is high: {rs.confidence_level:.2f}")

    dom_iters = dominant.get("dominant_iterations", 0)
    if dom_iters >= 2:
        bullets.append(f"Hypothesis has sustained dominance for {dom_iters} iteration(s)")

    if not bullets:
        bullets.append("No strong supporting factors identified from current evidence")

    return ExplanationSection(
        section_type=SectionType.SUPPORTING_EVIDENCE,
        title="Supporting Factors",
        bullet_points=bullets,
        referenced_fields=[
            "burst_detected", "mean_anomaly_score", "trend",
            "evidence_count", "source_diversity", "confidence_level",
            "dominant_iterations",
        ],
    )


def _build_contradicting_evidence(
    dominant: dict,
    hypotheses: list[dict],
    rs: SituationReasoningSnapshot,
    ts: SituationTemporalSnapshot,
) -> ExplanationSection:
    """CONTRADICTING_EVIDENCE: Factors weakening the dominant hypothesis."""
    bullets = []

    if rs.quiet_detected:
        bullets.append("Quiet period detected: reduces urgency of threat hypotheses")

    if rs.mean_anomaly_score < 0.3:
        bullets.append(f"Mean anomaly score is low: {rs.mean_anomaly_score:.2f}")

    if rs.trend.value == "deescalating":
        bullets.append("Trend is deescalating: activity may be subsiding")

    if rs.source_diversity <= 1:
        bullets.append(
            f"Source diversity is low ({rs.source_diversity}): "
            "single-source anomalies may be noise"
        )

    # UNKNOWN still has significant confidence
    unknown = next(
        (h for h in hypotheses if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID),
        None,
    )
    if unknown:
        unknown_conf = unknown.get("confidence", 0.0)
        dom_conf = dominant.get("confidence", 0.0)
        margin = dom_conf - unknown_conf
        if margin < 0.2:
            bullets.append(
                f"UNKNOWN hypothesis remains competitive (margin: {margin:.2f})"
            )

    if not bullets:
        bullets.append("No strong contradicting factors identified")

    return ExplanationSection(
        section_type=SectionType.CONTRADICTING_EVIDENCE,
        title="Contradicting Factors",
        bullet_points=bullets,
        referenced_fields=[
            "quiet_detected", "mean_anomaly_score", "trend",
            "source_diversity", "unknown_confidence",
        ],
    )


def _build_confidence_rationale(
    dominant: dict | None,
    hypotheses: list[dict],
    rs: SituationReasoningSnapshot,
    convergence: float,
    stability: float,
) -> ExplanationSection:
    """CONFIDENCE_RATIONALE: How confidence was computed."""
    bullets = []

    if dominant:
        conf = dominant.get("confidence", 0.0)
        vel = dominant.get("belief_velocity", 0.0)
        bullets.append(f"Dominant hypothesis confidence: {conf:.2f}")
        if abs(vel) > 0.01:
            direction = "increasing" if vel > 0 else "decreasing"
            bullets.append(f"Belief velocity: {vel:.3f} ({direction})")

    active_count = len([h for h in hypotheses if h.get("status") == "active"])
    pruned_count = len([h for h in hypotheses if h.get("status") == "pruned"])
    bullets.append(f"Active hypotheses: {active_count}, pruned: {pruned_count}")

    bullets.append(f"Convergence score: {convergence:.2f}")
    bullets.append(f"Belief stability: {stability:.2f}")

    bullets.append(
        f"Deterministic inputs: evidence={rs.evidence_count}, "
        f"anomaly={rs.mean_anomaly_score:.2f}, "
        f"diversity={rs.source_diversity}, "
        f"trend={rs.trend.value}"
    )

    return ExplanationSection(
        section_type=SectionType.CONFIDENCE_RATIONALE,
        title="Confidence Rationale",
        bullet_points=bullets,
        referenced_fields=[
            "confidence", "belief_velocity", "convergence_score",
            "belief_stability_score", "evidence_count",
            "mean_anomaly_score", "source_diversity", "trend",
        ],
    )


def _build_what_would_change(
    undecided: bool,
    dominant: dict | None,
    hypotheses: list[dict],
    rs: SituationReasoningSnapshot,
    ts: SituationTemporalSnapshot,
) -> ExplanationSection:
    """WHAT_WOULD_CHANGE_MY_MIND: What evidence would shift the system."""
    bullets = []

    if undecided:
        # What would resolve indecision?
        if rs.evidence_count < 5:
            bullets.append(
                f"More evidence needed: currently {rs.evidence_count}, "
                f"at least 5 signals would strengthen any hypothesis"
            )
        if rs.source_diversity < 3:
            bullets.append(
                f"Greater source diversity needed: currently {rs.source_diversity}, "
                f"3+ distinct sources would reduce noise risk"
            )
        bullets.append(
            "Sustained dominance of one hypothesis over UNKNOWN "
            "for 2+ consecutive iterations"
        )
    else:
        # What would overthrow the current belief?
        if rs.source_diversity <= 1:
            bullets.append(
                "Additional independent sources could confirm or deny "
                "the current hypothesis"
            )
        if rs.evidence_count < 10:
            bullets.append(
                f"Additional evidence (currently {rs.evidence_count}) "
                f"would increase confidence"
            )

        # What would make the system doubt?
        unknown = next(
            (h for h in hypotheses if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID),
            None,
        )
        if unknown:
            unknown_conf = unknown.get("confidence", 0.0)
            bullets.append(
                f"UNKNOWN confidence at {unknown_conf:.2f}: "
                f"rising above dominant would revert to undecided"
            )

        if not rs.quiet_detected:
            bullets.append(
                "A sustained quiet period would deescalate and reduce confidence"
            )

        if rs.trend.value != "deescalating":
            bullets.append(
                "A deescalating trend would weaken the current hypothesis"
            )

    if not bullets:
        bullets.append("No specific evidence gaps identified")

    return ExplanationSection(
        section_type=SectionType.WHAT_WOULD_CHANGE_MY_MIND,
        title="What Would Change the Assessment",
        bullet_points=bullets,
        referenced_fields=[
            "evidence_count", "source_diversity", "unknown_confidence",
            "quiet_detected", "trend", "dominant_iterations",
        ],
    )
