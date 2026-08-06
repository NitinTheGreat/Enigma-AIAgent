"""
Module: scenarios/scoring.py

Outcome metrics computed from the Level 6 per iteration run log.

Nothing here instruments anything. Every input is a record the run logger
already writes, which is the point of having built the logger first: the
metrics are derived from the same evidence trail a reader can inspect, not
from a parallel measurement that could disagree with it.

How a conclusion is matched to a ground truth. The reasoner emits free text,
so recognising which narrative a hypothesis names requires matching against
keywords. This is the weakest link in the scoring and its limits are stated
rather than hidden:

  - A hypothesis naming no known narrative is scored as a false conclusion,
    not as an abstention. The system committed to something; that it committed
    to something unrecognisable does not make it undecided.
  - A hypothesis naming both the true narrative and its rival is scored as
    false in an ambiguous scenario, because the scenario's correct answer is
    that the evidence does not separate them.
  - Keyword matching cannot detect a hypothesis that is right in substance but
    shares no vocabulary with the category. This inflates the false conclusion
    rate and deflates the correct conclusion rate, so the reported correct
    conclusion rate is a lower bound.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from scenarios.generator import (
    UNKNOWN_CONCLUSION,
    Regime,
    Scenario,
    scenario_id_from_entity,
)

UNKNOWN_HYPOTHESIS_ID = "UNKNOWN"


@dataclass
class SituationOutcome:
    """What the system did with one situation, against what it should have."""

    situation_id: str
    scenario_id: str
    regime: str
    expected_conclusion: str
    should_conclude: bool
    concluded: bool
    abstained: bool
    conclusion_text_hash: str | None
    matched_expected: bool
    matched_competitor: bool
    correct: bool
    false_conclusion: bool
    appropriate_abstention: bool
    inappropriate_abstention: bool
    premature: bool
    iterations: int
    evidence_count_at_termination: int
    sufficient_evidence_count: int
    termination_reason: str | None
    final_convergence: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable view."""
        return asdict(self)


@dataclass
class OutcomeMetrics:
    """The eight outcome metrics, plus the counts they were computed from."""

    situations: int = 0
    correct_conclusion_rate: float = 0.0
    false_conclusion_rate: float = 0.0
    abstention_rate: float = 0.0
    appropriate_abstention_rate: float = 0.0
    inappropriate_abstention_rate: float = 0.0
    premature_convergence_rate: float = 0.0
    single_iteration_conclusion_rate: float = 0.0
    mean_iterations_to_termination: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable view."""
        return asdict(self)


def load_run_log(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Group run log records by situation, ordered by iteration."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            grouped.setdefault(record["situation_id"], []).append(record)
    for records in grouped.values():
        records.sort(key=lambda r: r["iteration"])
    return grouped


def _dominant_hypothesis(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the record's dominant hypothesis object."""
    leader = record.get("dominant_hypothesis_id")
    for hypothesis in record.get("hypotheses", []):
        if hypothesis.get("hypothesis_id") == leader:
            return hypothesis
    return None


def _matches(text: str, keywords: Iterable[str]) -> bool:
    """Return whether any keyword appears in the text."""
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def score_situation(
    situation_id: str,
    records: list[dict[str, Any]],
    scenario: Scenario,
    descriptions: dict[str, str],
) -> SituationOutcome:
    """Score one situation against the ground truth of its scenario.

    Args:
        situation_id: The situation being scored.
        records: Its run log records, ordered by iteration.
        scenario: The scenario whose signals produced it.
        descriptions: Map from hypothesis text hash to the text it stands for,
            recovered from the run itself, because the log stores digests.

    Returns:
        What the system concluded and whether that was right.
    """
    final = records[-1]
    truth = scenario.ground_truth
    leader = _dominant_hypothesis(final)
    leader_id = final.get("dominant_hypothesis_id")

    abstained = leader_id == UNKNOWN_HYPOTHESIS_ID or leader is None
    concluded = not abstained

    text_hash = leader.get("text_hash") if leader else None
    text = descriptions.get(text_hash, "") if text_hash else ""

    matched_expected = concluded and _matches(text, truth.conclusion_keywords)
    matched_competitor = concluded and _matches(text, truth.competing_keywords)

    if truth.should_conclude:
        correct = concluded and matched_expected and not matched_competitor
    else:
        correct = abstained

    false_conclusion = concluded and not correct
    appropriate_abstention = abstained and not truth.should_conclude
    inappropriate_abstention = abstained and truth.should_conclude

    evidence_at_end = final.get("evidence_count") or 0
    premature = concluded and evidence_at_end < truth.sufficient_evidence_count

    return SituationOutcome(
        situation_id=situation_id,
        scenario_id=scenario.scenario_id,
        regime=truth.regime.value,
        expected_conclusion=truth.expected_conclusion,
        should_conclude=truth.should_conclude,
        concluded=concluded,
        abstained=abstained,
        conclusion_text_hash=text_hash,
        matched_expected=matched_expected,
        matched_competitor=matched_competitor,
        correct=correct,
        false_conclusion=false_conclusion,
        appropriate_abstention=appropriate_abstention,
        inappropriate_abstention=inappropriate_abstention,
        premature=premature,
        iterations=final.get("iteration") or len(records),
        evidence_count_at_termination=evidence_at_end,
        sufficient_evidence_count=truth.sufficient_evidence_count,
        termination_reason=final.get("termination_reason"),
        final_convergence=final.get("convergence_score") or 0.0,
    )


def aggregate(outcomes: list[SituationOutcome]) -> OutcomeMetrics:
    """Reduce per situation outcomes to the eight metrics.

    Denominators are chosen so each rate answers the question its name asks.
    Appropriate abstention is over situations whose truth says abstain, not
    over all situations, because a system that never abstains should score zero
    on it rather than being flattered by scenarios it was never asked about.
    """
    if not outcomes:
        return OutcomeMetrics()

    total = len(outcomes)
    concluded = [o for o in outcomes if o.concluded]
    abstained = [o for o in outcomes if o.abstained]
    should_abstain = [o for o in outcomes if not o.should_conclude]
    should_conclude = [o for o in outcomes if o.should_conclude]

    def rate(numerator: int, denominator: int) -> float:
        """Return a rounded ratio, zero when the denominator is empty."""
        return round(numerator / denominator, 4) if denominator else 0.0

    return OutcomeMetrics(
        situations=total,
        correct_conclusion_rate=rate(sum(1 for o in outcomes if o.correct), total),
        false_conclusion_rate=rate(sum(1 for o in outcomes if o.false_conclusion), total),
        abstention_rate=rate(len(abstained), total),
        appropriate_abstention_rate=rate(
            sum(1 for o in abstained if o.appropriate_abstention), len(should_abstain)
        ),
        inappropriate_abstention_rate=rate(
            sum(1 for o in abstained if o.inappropriate_abstention), len(should_conclude)
        ),
        premature_convergence_rate=rate(sum(1 for o in outcomes if o.premature), total),
        single_iteration_conclusion_rate=rate(
            sum(1 for o in concluded if o.iterations <= 1), total
        ),
        mean_iterations_to_termination=round(mean(o.iterations for o in outcomes), 4),
        counts={
            "situations": total,
            "concluded": len(concluded),
            "abstained": len(abstained),
            "truth_says_conclude": len(should_conclude),
            "truth_says_abstain": len(should_abstain),
            "correct": sum(1 for o in outcomes if o.correct),
            "false_conclusions": sum(1 for o in outcomes if o.false_conclusion),
            "premature": sum(1 for o in outcomes if o.premature),
        },
    )


def score_run(
    run_log_path: str | Path,
    scenarios: list[Scenario],
    situation_entities: dict[str, str],
    descriptions: dict[str, str],
) -> tuple[list[SituationOutcome], OutcomeMetrics, dict[str, OutcomeMetrics]]:
    """Score a whole validation run.

    Args:
        run_log_path: The Level 6 JSONL produced by the run.
        scenarios: The frozen suite the run was driven with.
        situation_entities: Map from situation id to one of its entity
            identifiers, which carries the scenario tag.
        descriptions: Map from hypothesis text hash to text.

    Returns:
        The per situation outcomes, the overall metrics, and the metrics broken
        down by regime.
    """
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    grouped = load_run_log(run_log_path)

    outcomes: list[SituationOutcome] = []
    for situation_id, records in grouped.items():
        entity = situation_entities.get(situation_id)
        if entity is None:
            continue
        scenario_id = scenario_id_from_entity(entity)
        scenario = by_id.get(scenario_id) if scenario_id else None
        if scenario is None:
            continue
        outcomes.append(
            score_situation(situation_id, records, scenario, descriptions)
        )

    per_regime = {
        regime.value: aggregate([o for o in outcomes if o.regime == regime.value])
        for regime in Regime
    }
    return outcomes, aggregate(outcomes), per_regime
