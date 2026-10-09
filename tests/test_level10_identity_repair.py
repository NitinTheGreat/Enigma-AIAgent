"""
Tests for the Level 10 stable hypothesis identity repair.

The properties that matter are that the repair is off by default, so every
earlier result reproduces unchanged; that it inherits only identity and the
dominant iteration counter, leaving description and confidence exactly as the
model produced them so the next prompt is unchanged; and that matching is one
to one, so no prior identity is handed to two successors.
"""

from __future__ import annotations

from enigma_reason.graph.builder import EpistemicControls
from enigma_reason.graph.nodes import (
    IDENTITY_MATCH_THRESHOLD,
    _identity_similarity,
    _inherit_identity,
    _new_hypothesis_dict,
)


def hypothesis(description: str, confidence: float = 0.4, dominant: int = 0) -> dict:
    """Build a hypothesis dict with a known dominant iteration count."""
    entry = _new_hypothesis_dict(description, confidence)
    entry["dominant_iterations"] = dominant
    return entry


def test_repair_is_off_by_default():
    """Every result from Levels 2 to 9 must reproduce unchanged."""
    assert EpistemicControls().stable_hypothesis_identity is False


def test_repair_appears_in_the_control_dictionary():
    """Manifests must record whether the repair was active."""
    controls = EpistemicControls(stable_hypothesis_identity=True)
    assert controls.as_dict()["stable_hypothesis_identity"] is True


def test_identical_text_inherits_identity_and_counter():
    """The commonest case in the logs is byte identical restatement."""
    prior = hypothesis("Coordinated reconnaissance from multiple sources", dominant=1)
    fresh = hypothesis("Coordinated reconnaissance from multiple sources")
    inherited = _inherit_identity([fresh], [prior])
    assert inherited == 1
    assert fresh["hypothesis_id"] == prior["hypothesis_id"]
    assert fresh["dominant_iterations"] == 1


def test_paraphrase_above_the_threshold_inherits():
    """A restatement with minor rewording is the same hypothesis."""
    prior = hypothesis("Coordinated reconnaissance activity from multiple sources.")
    fresh = hypothesis("Coordinated reconnaissance activity from multiple distinct sources.")
    assert _identity_similarity(prior["description"], fresh["description"]) >= IDENTITY_MATCH_THRESHOLD
    assert _inherit_identity([fresh], [prior]) == 1


def test_different_narrative_below_the_threshold_does_not_inherit():
    """Reconnaissance and exfiltration are not the same hypothesis."""
    prior = hypothesis("Sustained external reconnaissance from multiple IP addresses.")
    fresh = hypothesis("Large outbound data transfer to an unknown destination.")
    assert _identity_similarity(prior["description"], fresh["description"]) < IDENTITY_MATCH_THRESHOLD
    assert _inherit_identity([fresh], [prior]) == 0
    assert fresh["hypothesis_id"] != prior["hypothesis_id"]


def test_description_and_confidence_are_left_alone():
    """Inheriting identity must not change what the next prompt prints."""
    prior = hypothesis("Coordinated reconnaissance from multiple sources", confidence=0.9)
    fresh = hypothesis("Coordinated reconnaissance from multiple sources", confidence=0.31)
    _inherit_identity([fresh], [prior])
    assert fresh["description"] == "Coordinated reconnaissance from multiple sources"
    assert fresh["confidence"] == 0.31


def test_matching_is_one_to_one():
    """One prior identity must not be handed to two successors."""
    prior = hypothesis("Coordinated reconnaissance from multiple sources", dominant=2)
    first = hypothesis("Coordinated reconnaissance from multiple sources")
    second = hypothesis("Coordinated reconnaissance from multiple sources")
    inherited = _inherit_identity([first, second], [prior])
    assert inherited == 1
    assert sum(1 for h in (first, second) if h["hypothesis_id"] == prior["hypothesis_id"]) == 1


def test_best_match_wins_when_several_are_eligible():
    """Greedy assignment must take the closest pair first."""
    near = hypothesis("Coordinated reconnaissance activity from multiple sources")
    far = hypothesis("Coordinated reconnaissance activity from two sources")
    fresh = hypothesis("Coordinated reconnaissance activity from multiple sources")
    _inherit_identity([fresh], [near, far])
    assert fresh["hypothesis_id"] == near["hypothesis_id"]


def test_unknown_is_never_matched():
    """UNKNOWN already has a stable identity and must not be reassigned."""
    from enigma_reason.graph.nodes import UNKNOWN_HYPOTHESIS_ID

    prior = hypothesis("Insufficient evidence to determine")
    prior["hypothesis_id"] = UNKNOWN_HYPOTHESIS_ID
    fresh = hypothesis("Insufficient evidence to determine")
    assert _inherit_identity([fresh], [prior]) == 0


def test_pruned_priors_are_not_inherited_from():
    """A hypothesis the gate removed is not a live identity to continue."""
    prior = hypothesis("Coordinated reconnaissance from multiple sources", dominant=3)
    prior["status"] = "pruned"
    fresh = hypothesis("Coordinated reconnaissance from multiple sources")
    assert _inherit_identity([fresh], [prior]) == 0


def test_no_priors_is_a_no_op():
    """The first iteration has nothing to inherit from."""
    fresh = hypothesis("Coordinated reconnaissance from multiple sources")
    assert _inherit_identity([fresh], []) == 0


def test_inherited_pairs_record_their_similarity():
    """The spot check needs the score that justified each match."""
    prior = hypothesis("Coordinated reconnaissance activity from multiple sources.")
    fresh = hypothesis("Coordinated reconnaissance activity from multiple distinct sources.")
    _inherit_identity([fresh], [prior])
    assert fresh["identity_inherited"] is True
    assert 0.0 < fresh["identity_similarity"] <= 1.0
