"""Tests for Situation domain object."""

from datetime import timedelta

from enigma_reason.domain.situation import Situation
from enigma_reason.domain.signal import Signal
from enigma_reason.domain.enums import SignalType

from tests.test_signal import _valid_signal


class TestSituation:
    def _make_signal(self, **kw) -> Signal:
        return Signal.model_validate(_valid_signal(**kw))

    def test_new_situation_has_zero_evidence(self) -> None:
        sit = Situation()
        assert sit.evidence_count == 0

    def test_attach_evidence_increments_count(self) -> None:
        sit = Situation()
        sit.attach_evidence(self._make_signal())
        sit.attach_evidence(self._make_signal())
        assert sit.evidence_count == 2

    def test_evidence_returns_copy(self) -> None:
        sit = Situation()
        sit.attach_evidence(self._make_signal())
        evidence = sit.evidence
        evidence.clear()
        assert sit.evidence_count == 1  # internal list unaffected

    def test_is_expired_false_when_fresh(self) -> None:
        sit = Situation()
        assert not sit.is_expired(timedelta(minutes=30))

    def test_summary_contains_required_keys(self) -> None:
        sit = Situation()
        sit.attach_evidence(self._make_signal())
        s = sit.summary()
        assert "situation_id" in s
        assert "evidence_count" in s
        assert s["evidence_count"] == 1
