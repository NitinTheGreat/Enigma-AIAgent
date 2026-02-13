"""ExplanationBuilder — pure, deterministic explanation generator.

Accepts completed ReasoningState + Phase 2/4 snapshots and produces
an ExplanationSnapshot describing what the system believes and why.

Phase 6.1 adds:
- Evidence contribution scoring on SUPPORTING/CONTRADICTING sections
- COUNTERFACTUALS section with structured what-if projections
- TEMPORAL_EVOLUTION section with belief dynamics summary
- Integrity validation (fail-closed)

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
    ContributionDirection,
    Counterfactual,
    ExplanationSection,
    ExplanationSnapshot,
    SectionType,
    TemporalEvolution,
)
from enigma_reason.domain.hypothesis import UNKNOWN_HYPOTHESIS_ID
from enigma_reason.domain.reasoning import SituationReasoningSnapshot
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.graph.state import ReasoningState

logger = logging.getLogger(__name__)


# ── Known reasoning fields (for integrity validation) ────────────────────────

KNOWN_FIELDS: frozenset[str] = frozenset({
    "convergence_score", "iteration_count", "dominant_hypothesis",
    "evidence_count", "source_diversity", "confidence_spread",
    "belief_stability_score", "burst_detected", "mean_anomaly_score",
    "trend", "confidence_level", "quiet_detected", "unknown_confidence",
    "confidence", "belief_velocity", "dominant_iterations",
    "event_rate_per_minute", "active_duration_seconds",
    "undecided_iterations",
})


def build_explanation(
    reasoning_state: dict[str, Any],
    reasoning_snapshot: SituationReasoningSnapshot,
    temporal_snapshot: SituationTemporalSnapshot,
) -> ExplanationSnapshot:
    """Build a structured explanation from completed reasoning state.

    This is the public interface for Phase 6.  It reads the final
    ReasoningState (as a dict) plus the Phase 2 and Phase 4 snapshots,
    and produces an immutable ExplanationSnapshot.

    Phase 6.1: Adds counterfactuals, temporal evolution, contribution
    scores, and runs integrity validation before returning.

    Args:
        reasoning_state: Final ReasoningState dict from the graph runner.
        reasoning_snapshot: Phase 4 SituationReasoningSnapshot.
        temporal_snapshot: Phase 2 SituationTemporalSnapshot.

    Returns:
        Frozen ExplanationSnapshot with ordered explanation sections.

    Raises:
        ExplanationIntegrityError: If any section references unknown fields
            or contains unjustified content.
    """
    hypotheses = reasoning_state.get("hypotheses", [])
    convergence = reasoning_state.get("convergence_score", 0.0)
    stability = reasoning_state.get("belief_stability_score", 0.0)
    iteration_count = reasoning_state.get("iteration_count", 0)
    situation_id = reasoning_state.get("situation_id", "")
    undecided_iters = reasoning_state.get("undecided_iterations", 0)

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

    # Phase 6.1: New sections
    sections.append(_build_counterfactuals(
        is_undecided, dominant, hypotheses, reasoning_snapshot, temporal_snapshot,
    ))

    sections.append(_build_temporal_evolution_section(
        dominant, stability, undecided_iters,
    ))

    # Phase 6.1: Temporal evolution summary
    temporal_evo = _compute_temporal_evolution(
        dominant, stability, undecided_iters,
    )

    snapshot = ExplanationSnapshot(
        situation_id=situation_id,
        dominant_hypothesis_id=dominant_id,
        dominant_hypothesis_description=dominant_desc,
        dominant_confidence=round(dominant_conf, 4),
        convergence_score=round(convergence, 4),
        belief_stability_score=round(stability, 4),
        undecided=is_undecided,
        iteration_count=iteration_count,
        explanation_sections=sections,
        temporal_evolution=temporal_evo,
    )

    # Phase 6.1: Integrity validation (fail-closed)
    validate_explanation_integrity(snapshot)

    return snapshot


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

    if rs.evidence_count < 3:
        bullets.append(f"Evidence is sparse: only {rs.evidence_count} signal(s) observed")

    if rs.source_diversity <= 1:
        bullets.append(f"Source diversity is low: {rs.source_diversity} distinct source(s)")

    active = [h for h in hypotheses if h.get("status") == "active"
              and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID]
    if len(active) >= 2:
        confs = [h["confidence"] for h in active]
        spread = max(confs) - min(confs)
        if spread < 0.15:
            bullets.append(f"Hypothesis confidence spread is flat: {spread:.2f}")

    if stability < 0.5:
        bullets.append(f"Belief stability is low: {stability:.2f} (confidence still shifting)")

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
        contribution_direction=ContributionDirection.NEUTRAL,
    )


def _build_supporting_evidence(
    dominant: dict,
    rs: SituationReasoningSnapshot,
    ts: SituationTemporalSnapshot,
) -> ExplanationSection:
    """SUPPORTING_EVIDENCE: Factors reinforcing the dominant hypothesis."""
    bullets = []
    total_score = 0.0

    if rs.burst_detected:
        bullets.append("Burst activity detected: supports escalation-related hypotheses")
        total_score += 0.2

    if rs.mean_anomaly_score > 0.6:
        bullets.append(f"Mean anomaly score is elevated: {rs.mean_anomaly_score:.2f}")
        total_score += min(rs.mean_anomaly_score * 0.3, 0.3)

    if rs.trend.value == "escalating":
        bullets.append("Trend is escalating: consistent with sustained activity")
        total_score += 0.15

    if rs.evidence_count >= 5:
        bullets.append(f"Evidence volume is sufficient: {rs.evidence_count} signal(s)")
        total_score += 0.1

    if rs.source_diversity >= 3:
        bullets.append(f"Source diversity is high: {rs.source_diversity} distinct sources")
        total_score += 0.15

    if rs.confidence_level > 0.6:
        bullets.append(f"Deterministic confidence level is high: {rs.confidence_level:.2f}")
        total_score += 0.1

    dom_iters = dominant.get("dominant_iterations", 0)
    if dom_iters >= 2:
        bullets.append(f"Hypothesis has sustained dominance for {dom_iters} iteration(s)")
        total_score += 0.1

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
        contribution_score=round(min(total_score, 1.0), 4),
        contribution_direction=ContributionDirection.SUPPORTING,
    )


def _build_contradicting_evidence(
    dominant: dict,
    hypotheses: list[dict],
    rs: SituationReasoningSnapshot,
    ts: SituationTemporalSnapshot,
) -> ExplanationSection:
    """CONTRADICTING_EVIDENCE: Factors weakening the dominant hypothesis."""
    bullets = []
    total_score = 0.0

    if rs.quiet_detected:
        bullets.append("Quiet period detected: reduces urgency of threat hypotheses")
        total_score += 0.2

    if rs.mean_anomaly_score < 0.3:
        bullets.append(f"Mean anomaly score is low: {rs.mean_anomaly_score:.2f}")
        total_score += 0.15

    if rs.trend.value == "deescalating":
        bullets.append("Trend is deescalating: activity may be subsiding")
        total_score += 0.15

    if rs.source_diversity <= 1:
        bullets.append(
            f"Source diversity is low ({rs.source_diversity}): "
            "single-source anomalies may be noise"
        )
        total_score += 0.2

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
            total_score += 0.2

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
        contribution_score=round(min(total_score, 1.0), 4),
        contribution_direction=ContributionDirection.OPPOSING,
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


# ── Phase 6.1: Counterfactuals ──────────────────────────────────────────────


def _build_counterfactuals(
    undecided: bool,
    dominant: dict | None,
    hypotheses: list[dict],
    rs: SituationReasoningSnapshot,
    ts: SituationTemporalSnapshot,
) -> ExplanationSection:
    """COUNTERFACTUALS: Structured what-if projections from existing thresholds.

    Every counterfactual is derived from known deterministic rules in
    the evaluation and convergence nodes.  No speculation.
    """
    cfs: list[Counterfactual] = []
    bullets: list[str] = []

    # ── Evidence volume counterfactual ──────────────────────────────────
    if rs.evidence_count < 5:
        gap = 5 - rs.evidence_count
        # Sanity gate boosts UNKNOWN by +0.15 when evidence < 3
        # and +0.05 when < 5.  More evidence removes these boosts.
        delta = 0.10 if rs.evidence_count < 3 else 0.05
        cf = Counterfactual(
            missing_condition=f"{gap} more signal(s) observed (currently {rs.evidence_count})",
            expected_effect="UNKNOWN confidence penalty removed, competing hypotheses strengthened",
            confidence_delta_estimate=round(delta, 2),
        )
        cfs.append(cf)
        bullets.append(f"If {gap} more signal(s) observed: confidence delta +{delta:.2f}")

    # ── Source diversity counterfactual ──────────────────────────────────
    if rs.source_diversity < 3:
        gap = 3 - rs.source_diversity
        # Sanity gate boosts UNKNOWN by +0.1 when diversity <= 1
        # Convergence halved when diversity <= 1 + high anomaly
        delta = 0.15 if rs.source_diversity <= 1 else 0.08
        cf = Counterfactual(
            missing_condition=f"{gap} more independent source(s) (currently {rs.source_diversity})",
            expected_effect="Noise risk reduced, convergence penalty removed",
            confidence_delta_estimate=round(delta, 2),
        )
        cfs.append(cf)
        bullets.append(f"If {gap} more source(s): confidence delta +{delta:.2f}")

    # ── Burst persistence counterfactual ────────────────────────────────
    if not rs.burst_detected:
        cf = Counterfactual(
            missing_condition="Burst activity detected in signal pattern",
            expected_effect="Active hypotheses boosted by evaluation node",
            confidence_delta_estimate=0.10,
        )
        cfs.append(cf)
        bullets.append("If burst detected: confidence delta +0.10")

    # ── Sustained dominance counterfactual ──────────────────────────────
    if dominant and dominant.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID:
        dom_iters = dominant.get("dominant_iterations", 0)
        if dom_iters < 2:
            needed = 2 - dom_iters
            cf = Counterfactual(
                missing_condition=f"{needed} more iteration(s) of sustained dominance",
                expected_effect="Convergence persistence requirement met, convergence unlocked",
                confidence_delta_estimate=0.20,
            )
            cfs.append(cf)
            bullets.append(f"If {needed} more dominant iteration(s): convergence unlocked (+0.20)")

    # ── Quiet period counterfactual (for converged scenarios) ───────────
    if not undecided and not rs.quiet_detected:
        cf = Counterfactual(
            missing_condition="Sustained quiet period observed",
            expected_effect="Confidence reduced by evaluation node, deescalation signal",
            confidence_delta_estimate=-0.10,
        )
        cfs.append(cf)
        bullets.append("If quiet period detected: confidence delta -0.10")

    if not bullets:
        bullets.append("No actionable counterfactuals identified from current thresholds")

    return ExplanationSection(
        section_type=SectionType.COUNTERFACTUALS,
        title="Counterfactual Projections",
        bullet_points=bullets,
        referenced_fields=[
            "evidence_count", "source_diversity", "burst_detected",
            "dominant_iterations", "quiet_detected",
        ],
        counterfactuals=cfs,
    )


# ── Phase 6.1: Temporal Evolution ───────────────────────────────────────────


def _compute_temporal_evolution(
    dominant: dict | None,
    stability: float,
    undecided_iters: int,
) -> TemporalEvolution:
    """Compute the temporal evolution summary from belief dynamics."""
    vel = dominant.get("belief_velocity", 0.0) if dominant else 0.0

    # Confidence trend
    if vel > 0.02:
        trend = "rising"
    elif vel < -0.02:
        trend = "falling"
    else:
        trend = "flat"

    # Velocity magnitude
    abs_vel = abs(vel)
    if abs_vel < 0.03:
        vel_summary = "slow"
    elif abs_vel < 0.10:
        vel_summary = "moderate"
    else:
        vel_summary = "fast"

    # Stability label
    if stability >= 0.7:
        stab_label = "stable"
    elif stability >= 0.4:
        stab_label = "transitioning"
    else:
        stab_label = "volatile"

    return TemporalEvolution(
        confidence_trend=trend,
        belief_velocity_summary=vel_summary,
        undecided_duration=undecided_iters,
        stability_label=stab_label,
    )


def _build_temporal_evolution_section(
    dominant: dict | None,
    stability: float,
    undecided_iters: int,
) -> ExplanationSection:
    """TEMPORAL_EVOLUTION: Describe how beliefs changed over iterations."""
    evo = _compute_temporal_evolution(dominant, stability, undecided_iters)
    bullets = [
        f"Confidence trend: {evo.confidence_trend}",
        f"Belief velocity: {evo.belief_velocity_summary}",
        f"Stability: {evo.stability_label}",
    ]

    if undecided_iters > 0:
        bullets.append(f"Undecided for {undecided_iters} iteration(s)")

    vel = dominant.get("belief_velocity", 0.0) if dominant else 0.0
    if abs(vel) > 0.01:
        direction = "accelerating" if vel > 0 else "decelerating"
        bullets.append(f"Belief is {direction} (velocity: {vel:.3f})")

    return ExplanationSection(
        section_type=SectionType.TEMPORAL_EVOLUTION,
        title="Temporal Belief Evolution",
        bullet_points=bullets,
        referenced_fields=[
            "belief_velocity", "belief_stability_score",
            "undecided_iterations",
        ],
    )


# ── Phase 6.1: Integrity Validation ─────────────────────────────────────────


class ExplanationIntegrityError(Exception):
    """Raised when an explanation fails integrity validation.

    This error means the explanation contains references to unknown
    fields or unjustified content — a sign of builder regression.
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            f"Explanation integrity check failed: {'; '.join(violations)}"
        )


def validate_explanation_integrity(snapshot: ExplanationSnapshot) -> None:
    """Validate that every section references only known reasoning fields.

    Fail-closed: if any violation is found, raise ExplanationIntegrityError.
    This protects against future regression and LLM misuse.
    """
    violations: list[str] = []

    for section in snapshot.explanation_sections:
        # Check referenced fields are all known
        for field in section.referenced_fields:
            if field not in KNOWN_FIELDS:
                violations.append(
                    f"Section '{section.section_type.value}' references "
                    f"unknown field '{field}'"
                )

        # Check section has justification (non-empty bullets)
        if not section.bullet_points:
            violations.append(
                f"Section '{section.section_type.value}' has no bullet points"
            )

        # Check contribution scores are bounded
        if section.contribution_score is not None:
            if not (0.0 <= section.contribution_score <= 1.0):
                violations.append(
                    f"Section '{section.section_type.value}' has "
                    f"out-of-bounds contribution score: {section.contribution_score}"
                )

        # Check counterfactuals have valid deltas
        for cf in section.counterfactuals:
            if not (-1.0 <= cf.confidence_delta_estimate <= 1.0):
                violations.append(
                    f"Counterfactual '{cf.missing_condition[:50]}' has "
                    f"out-of-bounds delta: {cf.confidence_delta_estimate}"
                )

    if violations:
        raise ExplanationIntegrityError(violations)
