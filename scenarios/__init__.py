"""
Module: scenarios/__init__.py

Synthetic scenarios carrying injected ground truth.

Nothing downstream of the classifier could be evaluated before this package
existed, because the corpus has no situation level label. Level 5 showed the
cost precisely: rule only accuracy came out at 1.0 because entity assignment
was uncorrelated with attack class, so every situation was a uniform random
sample of the corpus and the metric was asking whether the most common class
equals the most common class.

Because the scenario is generated, the conclusion a competent reasoner should
reach is known, including the case where that conclusion is that the evidence
does not support one.
"""

from scenarios.generator import (
    GroundTruth,
    Regime,
    Scenario,
    ScenarioGenerator,
    ScenarioParameters,
    generate_suite,
)
from scenarios.scoring import (
    OutcomeMetrics,
    SituationOutcome,
    score_run,
    score_situation,
)

__all__ = [
    "GroundTruth",
    "Regime",
    "Scenario",
    "ScenarioGenerator",
    "ScenarioParameters",
    "generate_suite",
    "OutcomeMetrics",
    "SituationOutcome",
    "score_run",
    "score_situation",
]
