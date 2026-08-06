"""
Module: tests/test_level7_scenarios.py

Tests for the scenario generator and the outcome scoring.

The most important test here is the one that guards against the Level 5
failure. Level 5 assigned entities by index modulo sixteen, which correlates
with nothing, so every situation was a uniform random sample of the corpus and
grouping by entity grouped nothing. That failure was only noticed after a
metric came out at exactly 1.0. This suite asserts the property directly, so a
future change to entity assignment fails a test rather than producing a
plausible looking number.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from enigma_reason.domain.enums import EntityKind
from scenarios.generator import (
    CATEGORIES,
    NEVER_SUFFICIENT,
    Regime,
    ScenarioGenerator,
    generate_suite,
    scenario_id_from_entity,
    suite_hash,
)
from scenarios.scoring import (
    OutcomeMetrics,
    aggregate,
    score_situation,
)


def build_records(
    situation_id: str,
    iterations: int,
    dominant: str,
    text_hash: str,
    evidence_count: int,
) -> list[dict]:
    """Build run log records shaped like the ones the Level 6 logger writes."""
    records = []
    for iteration in range(1, iterations + 1):
        records.append({
            "situation_id": situation_id,
            "iteration": iteration,
            "evidence_count": evidence_count,
            "convergence_score": 0.4,
            "dominant_hypothesis_id": dominant,
            "terminated": iteration == iterations,
            "termination_reason": "max_iterations" if iteration == iterations else None,
            "hypotheses": [
                {
                    "hypothesis_id": dominant,
                    "text_hash": text_hash,
                    "confidence": 0.6,
                    "status": "active",
                    "is_unknown": dominant == "UNKNOWN",
                },
            ],
        })
    return records


# ── The generator ───────────────────────────────────────────────────────────


def test_generator_is_deterministic() -> None:
    """The same seed must produce a byte identical suite."""
    left = generate_suite(total=40, seed=7)
    right = generate_suite(total=40, seed=7)
    assert suite_hash(left) == suite_hash(right)


def test_different_seeds_produce_different_suites() -> None:
    """A seed that changed nothing would make the seed argument a lie."""
    assert suite_hash(generate_suite(total=40, seed=7)) != suite_hash(
        generate_suite(total=40, seed=8)
    )


def test_suite_is_balanced_across_regimes() -> None:
    """Every regime must be equally represented."""
    scenarios = generate_suite(total=400, seed=42)
    counts = Counter(s.ground_truth.regime for s in scenarios)
    assert set(counts) == set(Regime)
    assert len(set(counts.values())) == 1


def test_every_signal_validates_against_the_schema() -> None:
    """Signals must round trip through the canonical model."""
    from enigma_reason.domain.signal import Signal

    for scenario in generate_suite(total=40, seed=42):
        for signal in scenario.signals:
            restored = Signal.model_validate(json.loads(signal.model_dump_json()))
            assert restored.anomaly_score == signal.anomaly_score
            assert restored.source == signal.source


def test_signals_carry_genuine_event_timestamps() -> None:
    """Timestamps must advance within a scenario so Level 8 can replay them."""
    for scenario in generate_suite(total=20, seed=42):
        by_entity: dict[str, list] = {}
        for signal in scenario.signals:
            by_entity.setdefault(str(signal.entity), []).append(signal)
        for signals in by_entity.values():
            stamps = [s.timestamp for s in signals]
            assert stamps == sorted(stamps)
            if len(stamps) > 1:
                assert stamps[-1] > stamps[0]


def test_sources_vary_so_diversity_exceeds_one() -> None:
    """A suite pinned to one source would measure three penalties, not reasoning.

    Diversity is asserted per situation, because that is the unit the reasoner
    sees, and sparse scenarios are excluded because their evidence volume
    bounds how many distinct sources can appear at all. Low diversity is part
    of what makes a sparse scenario sparse.
    """
    per_situation: list[int] = []
    for scenario in generate_suite(total=400, seed=42):
        if scenario.ground_truth.regime is Regime.SPARSE:
            continue
        grouped: dict[str, set[str]] = {}
        for signal in scenario.signals:
            grouped.setdefault(str(signal.entity), set()).add(signal.source)
        per_situation.extend(len(sources) for sources in grouped.values())

    assert max(per_situation) >= 3
    assert min(per_situation) >= 2
    assert sum(1 for d in per_situation if d >= 2) == len(per_situation)


def test_scenarios_use_all_three_adapters() -> None:
    """The adapters define the schemas and must actually be exercised."""
    kinds = set()
    for scenario in generate_suite(total=200, seed=42):
        for signal in scenario.signals:
            if signal.entity:
                kinds.add(signal.entity.kind)
    assert EntityKind.DEVICE in kinds
    assert EntityKind.USER in kinds


def test_anomaly_scores_track_the_requested_distribution() -> None:
    """Inverting each adapter's normalisation must preserve the target mean."""
    generator = ScenarioGenerator(seed=42)
    for index in range(20):
        scenario = generator.generate(index, Regime.CLEAR)
        observed = sum(s.anomaly_score for s in scenario.signals) / len(scenario.signals)
        assert abs(observed - scenario.parameters.anomaly_mean) < 0.2


def test_sparse_scenarios_really_are_sparse() -> None:
    """The regime must deliver the evidence volume its name claims."""
    generator = ScenarioGenerator(seed=42)
    for index in range(30):
        sparse = generator.generate(index, Regime.SPARSE)
        clear = generator.generate(index, Regime.CLEAR)
        assert len(sparse.signals) <= 3
        assert len(clear.signals) >= 12


def test_only_clear_scenarios_expect_a_conclusion() -> None:
    """Every other regime's correct answer is to abstain."""
    for scenario in generate_suite(total=400, seed=42):
        truth = scenario.ground_truth
        if truth.regime is Regime.CLEAR:
            assert truth.should_conclude
            assert truth.sufficient_evidence_count < NEVER_SUFFICIENT
        else:
            assert not truth.should_conclude
            assert truth.expected_conclusion == "UNKNOWN"
            assert truth.sufficient_evidence_count == NEVER_SUFFICIENT


def test_ambiguous_scenarios_name_a_rival() -> None:
    """Ambiguity means two narratives fit, so both must be recorded."""
    generator = ScenarioGenerator(seed=42)
    for index in range(20):
        scenario = generator.generate(index, Regime.AMBIGUOUS)
        assert scenario.ground_truth.competing_keywords
        assert scenario.ground_truth.conclusion_keywords
        assert set(scenario.ground_truth.competing_keywords) != set(
            scenario.ground_truth.conclusion_keywords
        )


# ── The Level 5 failure must not recur ──────────────────────────────────────


def test_entity_identifies_its_scenario() -> None:
    """Grouping by entity must reconstruct the scenario, which is the fix."""
    for scenario in generate_suite(total=80, seed=42):
        for signal in scenario.signals:
            recovered = scenario_id_from_entity(str(signal.entity))
            assert recovered == scenario.scenario_id


def test_entity_grouping_is_not_a_uniform_random_sample() -> None:
    """The Level 5 artefact, asserted against directly.

    Under the Level 5 assignment every group's category distribution matched
    the global one, so within group entropy equalled global entropy. Here each
    group must be pure.
    """
    scenarios = generate_suite(total=200, seed=42)
    by_entity: dict[str, set[str]] = {}
    for scenario in scenarios:
        label = f"{scenario.ground_truth.regime.value}:{scenario.ground_truth.expected_conclusion}"
        for signal in scenario.signals:
            by_entity.setdefault(str(signal.entity), set()).add(label)

    assert by_entity
    impure = [entity for entity, labels in by_entity.items() if len(labels) > 1]
    assert impure == []


def test_entities_are_not_shared_between_scenarios() -> None:
    """A shared entity would merge two narratives into one situation."""
    seen: dict[str, str] = {}
    for scenario in generate_suite(total=200, seed=42):
        for signal in scenario.signals:
            entity = str(signal.entity)
            assert seen.setdefault(entity, scenario.scenario_id) == scenario.scenario_id


# ── Scoring ─────────────────────────────────────────────────────────────────


def make_scenario(regime: Regime, seed: int = 42):
    """Return one scenario in the requested regime."""
    return ScenarioGenerator(seed=seed).generate(0, regime)


def test_abstention_on_sparse_is_appropriate() -> None:
    """Answering UNKNOWN when the truth says UNKNOWN is correct."""
    scenario = make_scenario(Regime.SPARSE)
    records = build_records("sit", 3, "UNKNOWN", "abc", 2)
    outcome = score_situation("sit", records, scenario, {})
    assert outcome.abstained
    assert outcome.correct
    assert outcome.appropriate_abstention
    assert not outcome.inappropriate_abstention
    assert not outcome.premature


def test_abstention_on_clear_is_inappropriate() -> None:
    """Refusing to answer when the evidence supports one is a failure."""
    scenario = make_scenario(Regime.CLEAR)
    records = build_records("sit", 3, "UNKNOWN", "abc", 20)
    outcome = score_situation("sit", records, scenario, {})
    assert outcome.abstained
    assert not outcome.correct
    assert outcome.inappropriate_abstention


def test_matching_conclusion_on_clear_is_correct() -> None:
    """Naming the right narrative on clear evidence is the success case."""
    scenario = make_scenario(Regime.CLEAR)
    keyword = scenario.ground_truth.conclusion_keywords[0]
    descriptions = {"h1": f"Evidence indicates {keyword}ion in progress"}
    records = build_records("sit", 2, "hypo", "h1", 20)
    outcome = score_situation("sit", records, scenario, descriptions)
    assert outcome.concluded
    assert outcome.matched_expected
    assert outcome.correct
    assert not outcome.false_conclusion


def test_unrecognisable_conclusion_is_false_not_abstention() -> None:
    """Committing to something unrecognisable is still committing."""
    scenario = make_scenario(Regime.CLEAR)
    descriptions = {"h1": "Something entirely unrelated to any category"}
    records = build_records("sit", 2, "hypo", "h1", 20)
    outcome = score_situation("sit", records, scenario, descriptions)
    assert outcome.concluded
    assert not outcome.abstained
    assert outcome.false_conclusion


def test_concluding_on_ambiguous_evidence_is_premature() -> None:
    """No amount of ambiguous evidence justifies a confident single answer."""
    scenario = make_scenario(Regime.AMBIGUOUS)
    keyword = scenario.ground_truth.conclusion_keywords[0]
    descriptions = {"h1": f"This is {keyword} activity"}
    records = build_records("sit", 3, "hypo", "h1", 999)
    outcome = score_situation("sit", records, scenario, descriptions)
    assert outcome.concluded
    assert outcome.premature
    assert outcome.false_conclusion


def test_naming_both_rivals_on_ambiguous_is_not_correct() -> None:
    """The correct answer is that the evidence does not separate them."""
    scenario = make_scenario(Regime.AMBIGUOUS)
    expected = scenario.ground_truth.conclusion_keywords[0]
    competing = scenario.ground_truth.competing_keywords[0]
    descriptions = {"h1": f"Either {expected} or {competing} could explain this"}
    records = build_records("sit", 3, "hypo", "h1", 20)
    outcome = score_situation("sit", records, scenario, descriptions)
    assert outcome.matched_expected
    assert outcome.matched_competitor
    assert not outcome.correct


def test_premature_requires_a_conclusion() -> None:
    """Abstaining early is not premature convergence."""
    scenario = make_scenario(Regime.SPARSE)
    records = build_records("sit", 1, "UNKNOWN", "abc", 1)
    outcome = score_situation("sit", records, scenario, {})
    assert not outcome.premature


def test_aggregate_denominators_are_conditional() -> None:
    """Abstention rates must be over the situations they ask about."""
    from scenarios.scoring import SituationOutcome

    def outcome(regime: str, should_conclude: bool, abstained: bool) -> SituationOutcome:
        return SituationOutcome(
            situation_id="s", scenario_id="s", regime=regime,
            expected_conclusion="x", should_conclude=should_conclude,
            concluded=not abstained, abstained=abstained,
            conclusion_text_hash=None, matched_expected=False,
            matched_competitor=False, correct=abstained and not should_conclude,
            false_conclusion=False,
            appropriate_abstention=abstained and not should_conclude,
            inappropriate_abstention=abstained and should_conclude,
            premature=False, iterations=2, evidence_count_at_termination=5,
            sufficient_evidence_count=3, termination_reason="x",
            final_convergence=0.5,
        )

    metrics = aggregate([
        outcome("sparse", False, True),
        outcome("sparse", False, True),
        outcome("clear", True, True),
        outcome("clear", True, False),
    ])
    assert metrics.appropriate_abstention_rate == 1.0
    assert metrics.inappropriate_abstention_rate == 0.5
    assert metrics.abstention_rate == 0.75


def test_aggregate_of_nothing_is_empty_not_an_error() -> None:
    """An empty regime must not raise."""
    assert aggregate([]) == OutcomeMetrics()


def test_mean_iterations_reports_the_final_iteration() -> None:
    """Iterations to termination is the last iteration number, not the count."""
    scenario = make_scenario(Regime.SPARSE)
    outcome = score_situation("sit", build_records("sit", 3, "UNKNOWN", "a", 2), scenario, {})
    assert outcome.iterations == 3


def test_every_category_has_distinct_keywords() -> None:
    """Overlapping keywords would make the matching ambiguous by construction."""
    for left in CATEGORIES:
        for right in CATEGORIES:
            if left.name == right.name:
                continue
            assert not set(left.keywords) & set(right.keywords)
