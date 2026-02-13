"""Smoke tests for Pydantic models."""

from uuid import uuid4

from enigma_reason.models.decision import ActionType, DecisionOutput
from enigma_reason.models.signal import IncomingSignal, SignalSeverity
from enigma_reason.models.state import SituationState, ThreatLevel


def test_incoming_signal_validates() -> None:
    signal = IncomingSignal(
        signal_id=uuid4(),
        source="camera-01",
        signal_type="intrusion",
        severity=SignalSeverity.HIGH,
        confidence=0.92,
    )
    assert signal.severity == SignalSeverity.HIGH


def test_situation_state_defaults() -> None:
    state = SituationState()
    assert state.threat_level == ThreatLevel.NONE
    assert state.active_signals == []


def test_decision_output_validates() -> None:
    decision = DecisionOutput(
        action=ActionType.ALERT,
        summary="Perimeter breach detected",
        confidence=0.88,
    )
    assert decision.action == ActionType.ALERT
