"""ReasoningEngine — deterministic confidence and trend computation.

Design principles:
    1. Pure function: accepts a Situation, returns a SituationReasoningSnapshot.
    2. No side effects, no state mutation, no I/O.
    3. No LLMs, no ML inference.
    4. All weights are explicit and configurable.
    5. This is the bridge to LangGraph in future phases.

Confidence formula:
    confidence = clamp(
        w_evidence  * evidence_contribution
      + w_rate      * rate_contribution
      + w_diversity * diversity_contribution
      + w_anomaly   * anomaly_contribution
      + w_burst     * burst_bonus
    , 0.0, 1.0)

    Where each contribution is a normalised [0, 1] value:
    - evidence_contribution = min(evidence_count / evidence_saturation, 1.0)
    - rate_contribution     = min(event_rate / rate_saturation, 1.0)
    - diversity_contribution = min(source_count / diversity_saturation, 1.0)
    - anomaly_contribution  = mean_anomaly_score  (already 0–1)
    - burst_bonus           = 1.0 if bursting else 0.0

Trend detection:
    Uses the recent vs overall event rate comparison and burst/quiet state:
    - ESCALATING:    burst_detected OR recent_rate > overall_rate * rate_rise_factor
    - DEESCALATING:  quiet_detected OR (recent_rate < overall_rate / rate_fall_factor AND has data)
    - STABLE:        everything else
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from enigma_reason.domain.reasoning import SituationReasoningSnapshot, Trend
from enigma_reason.domain.situation import Situation


@dataclass(frozen=True)
class ConfidenceWeights:
    """Configurable weights for the confidence formula.

    All weights should sum to roughly 1.0 for intuitive results,
    but this is not enforced — they are normalised internally.
    """

    evidence: float = 0.25
    rate: float = 0.15
    diversity: float = 0.20
    anomaly: float = 0.30
    burst: float = 0.10

    # Saturation points — how many items/rate/sources to hit 1.0
    evidence_saturation: float = 10.0
    rate_saturation: float = 10.0  # events/min
    diversity_saturation: float = 3.0  # distinct sources


@dataclass(frozen=True)
class TrendConfig:
    """Configurable thresholds for trend detection."""

    # How many times the recent rate must exceed overall for escalation
    rate_rise_factor: float = 1.5
    # How many times below overall rate counts as deescalation
    rate_fall_factor: float = 2.0
    # Recent window for rate comparison
    recent_count: int = 3


class ReasoningEngine:
    """Deterministic reasoning over situations.

    This engine is stateless: it accepts a Situation and produces
    a SituationReasoningSnapshot.  It never mutates the situation.
    """

    def __init__(
        self,
        weights: ConfidenceWeights | None = None,
        trend_config: TrendConfig | None = None,
        burst_factor: float = 3.0,
        burst_recent_count: int = 3,
        quiet_window: timedelta = timedelta(minutes=5),
    ) -> None:
        self._weights = weights or ConfidenceWeights()
        self._trend_config = trend_config or TrendConfig()
        self._burst_factor = burst_factor
        self._burst_recent_count = burst_recent_count
        self._quiet_window = quiet_window

    # ── Public API ───────────────────────────────────────────────────────

    def evaluate(self, situation: Situation) -> SituationReasoningSnapshot:
        """Produce a reasoning snapshot for the given situation.

        Returns an immutable snapshot with confidence and trend derived
        deterministically from evidence and temporal facts.
        """
        evidence_count = situation.evidence_count
        event_rate = situation.event_rate
        burst = situation.is_bursting(self._burst_factor, self._burst_recent_count)
        quiet = situation.is_quiet(self._quiet_window)
        sources = self._source_diversity(situation)
        mean_anomaly = self._mean_anomaly_score(situation)

        confidence = self._compute_confidence(
            evidence_count=evidence_count,
            event_rate=event_rate,
            source_diversity=sources,
            mean_anomaly=mean_anomaly,
            burst=burst,
        )

        trend = self._detect_trend(situation, burst=burst, quiet=quiet)

        return SituationReasoningSnapshot(
            situation_id=str(situation.situation_id),
            evidence_count=evidence_count,
            event_rate=event_rate,
            burst_detected=burst,
            quiet_detected=quiet,
            confidence_level=round(confidence, 4),
            trend=trend,
            source_diversity=sources,
            mean_anomaly_score=round(mean_anomaly, 4),
        )

    # ── Confidence ───────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        evidence_count: int,
        event_rate: float,
        source_diversity: int,
        mean_anomaly: float,
        burst: bool,
    ) -> float:
        w = self._weights

        evidence_c = min(evidence_count / max(w.evidence_saturation, 1.0), 1.0)
        rate_c = min(event_rate / max(w.rate_saturation, 1.0), 1.0)
        diversity_c = min(source_diversity / max(w.diversity_saturation, 1.0), 1.0)
        anomaly_c = mean_anomaly  # already [0, 1]
        burst_c = 1.0 if burst else 0.0

        raw = (
            w.evidence * evidence_c
            + w.rate * rate_c
            + w.diversity * diversity_c
            + w.anomaly * anomaly_c
            + w.burst * burst_c
        )

        return max(0.0, min(raw, 1.0))

    # ── Trend detection ──────────────────────────────────────────────────

    def _detect_trend(
        self,
        situation: Situation,
        burst: bool,
        quiet: bool,
    ) -> Trend:
        # No evidence → nothing to reason about
        if situation.evidence_count == 0:
            return Trend.STABLE

        # Burst always implies escalation
        if burst:
            return Trend.ESCALATING

        # Quiet implies deescalation only if we have evidence
        if quiet and situation.evidence_count > 0:
            return Trend.DEESCALATING

        # Compare recent intervals vs overall to detect rate change
        intervals = situation.event_intervals
        rc = self._trend_config.recent_count

        if len(intervals) < rc:
            return Trend.STABLE

        overall_mean = sum(intervals) / len(intervals)
        recent_mean = sum(intervals[-rc:]) / rc

        if overall_mean == 0.0:
            return Trend.STABLE

        # Shorter recent intervals = faster rate = escalating
        if recent_mean < (overall_mean / self._trend_config.rate_rise_factor):
            return Trend.ESCALATING

        # Longer recent intervals = slower rate = deescalating
        if recent_mean > (overall_mean * self._trend_config.rate_fall_factor):
            return Trend.DEESCALATING

        return Trend.STABLE

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _source_diversity(situation: Situation) -> int:
        """Count distinct signal sources."""
        if not situation.evidence:
            return 0
        return len({sig.source for sig in situation.evidence})

    @staticmethod
    def _mean_anomaly_score(situation: Situation) -> float:
        """Average anomaly score across all evidence."""
        evidence = situation.evidence
        if not evidence:
            return 0.0
        return sum(sig.anomaly_score for sig in evidence) / len(evidence)
