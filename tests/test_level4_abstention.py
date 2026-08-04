"""
Module: tests/test_level4_abstention.py

Regression tests for the Level 4 abstention path.

The sensor may decline to assign a class when its calibrated confidence falls
below the validation-fitted threshold. An abstained signal is still evidence
that something occurred, so it must reach the reasoner intact rather than being
dropped or silently converted into a confident label.

These tests fail against the pre-Level-4 code, where Signal carried no abstained
field and the reasoning snapshot had no way to express that a situation rests on
detections the sensor would not stand behind.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from enigma_reason.adapters.unsw_threat import UNSWThreatAdapter
from enigma_reason.core.reasoning_engine import ReasoningEngine
from enigma_reason.domain.enums import SignalType
from enigma_reason.domain.signal import Signal
from enigma_reason.domain.situation import Situation
from enigma_reason.foundation.clock import ReplayClock
from enigma_reason.store.correlation import EntityCorrelation
from enigma_reason.store.situation_store import SituationStore

_BASE = datetime(2015, 1, 22, 11, 49, 37, tzinfo=timezone.utc)


def _sensor_payload(
    signal_type: str = "unknown",
    abstained: bool = True,
    calibrated_confidence: float = 0.41,
    device: str = "synthetic-device-01",
    offset_seconds: int = 0,
) -> dict:
    """Build a sensor payload in the exact shape main.py emits."""
    return {
        "inputs_for_xai_model": {
            "signal_id": f"00000000-0000-4000-8000-{offset_seconds:012d}",
            "timestamp": (_BASE + timedelta(seconds=offset_seconds)).isoformat(),
            "signal_type": signal_type,
            "entity": {"device": device, "user": "network_admin", "location": "rack"},
            "abstained": abstained,
            "calibrated_confidence": calibrated_confidence,
            "anomaly_score": 0.77,
            "predicted_class_confidence": 0.41,
            "predictive_entropy": 0.63,
            "features": ["dur", "sbytes"],
            "source": "unsw-threat-detector",
        }
    }


class TestSignalCarriesAbstention:
    def test_signal_defaults_to_not_abstained(self) -> None:
        signal = Signal.model_validate(
            {
                "signal_id": "00000000-0000-4000-8000-000000000001",
                "timestamp": _BASE.isoformat(),
                "signal_type": "generic",
                "anomaly_score": 0.9,
                "confidence": 0.9,
                "source": "unsw-threat-detector",
            }
        )
        assert signal.abstained is False
        assert signal.calibrated_confidence is None

    def test_adapter_preserves_abstention(self) -> None:
        signal = UNSWThreatAdapter().adapt(_sensor_payload())
        assert signal.abstained is True
        assert signal.calibrated_confidence == pytest.approx(0.41)
        assert signal.signal_type is SignalType.UNKNOWN

    def test_adapter_preserves_a_confident_classification(self) -> None:
        signal = UNSWThreatAdapter().adapt(
            _sensor_payload(signal_type="generic", abstained=False, calibrated_confidence=0.97)
        )
        assert signal.abstained is False
        assert signal.signal_type is SignalType.GENERIC

    def test_calibrated_confidence_is_clamped(self) -> None:
        signal = UNSWThreatAdapter().adapt(_sensor_payload(calibrated_confidence=1.4))
        assert signal.calibrated_confidence == pytest.approx(1.0)

    def test_absent_abstention_field_defaults_false(self) -> None:
        payload = _sensor_payload()
        del payload["inputs_for_xai_model"]["abstained"]
        del payload["inputs_for_xai_model"]["calibrated_confidence"]
        signal = UNSWThreatAdapter().adapt(payload)
        assert signal.abstained is False
        assert signal.calibrated_confidence is None


class TestReasonerSurfacesAbstention:
    def test_snapshot_reports_zero_when_nothing_abstained(self) -> None:
        clock = ReplayClock()
        situation = Situation(clock=clock)
        for offset in range(4):
            situation.attach_evidence(
                UNSWThreatAdapter().adapt(
                    _sensor_payload(
                        signal_type="generic", abstained=False, offset_seconds=offset
                    )
                )
            )
        snapshot = ReasoningEngine().evaluate(situation)
        assert snapshot.abstained_evidence_count == 0
        assert snapshot.abstained_fraction == 0.0

    def test_snapshot_counts_abstained_evidence(self) -> None:
        clock = ReplayClock()
        situation = Situation(clock=clock)
        for offset in range(4):
            situation.attach_evidence(
                UNSWThreatAdapter().adapt(_sensor_payload(offset_seconds=offset))
            )
        snapshot = ReasoningEngine().evaluate(situation)
        assert snapshot.abstained_evidence_count == 4
        assert snapshot.abstained_fraction == pytest.approx(1.0)

    def test_snapshot_reports_a_mixed_fraction(self) -> None:
        clock = ReplayClock()
        situation = Situation(clock=clock)
        for offset in range(3):
            situation.attach_evidence(
                UNSWThreatAdapter().adapt(_sensor_payload(offset_seconds=offset))
            )
        for offset in range(3, 7):
            situation.attach_evidence(
                UNSWThreatAdapter().adapt(
                    _sensor_payload(
                        signal_type="generic", abstained=False, offset_seconds=offset
                    )
                )
            )
        snapshot = ReasoningEngine().evaluate(situation)
        assert snapshot.abstained_evidence_count == 3
        assert snapshot.abstained_fraction == pytest.approx(3 / 7, abs=1e-4)


class TestAbstentionReachesTheStore:
    @pytest.mark.asyncio
    async def test_abstained_signal_is_ingested_not_dropped(self) -> None:
        engine = ReasoningEngine()
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
            correlation=EntityCorrelation(),
            reasoning_engine=engine,
        )
        signal = UNSWThreatAdapter().adapt(_sensor_payload())
        situation = await store.ingest(signal)

        assert situation.evidence_count == 1
        assert situation.evidence[0].abstained is True

        snapshot = engine.evaluate(situation)
        assert snapshot.abstained_evidence_count == 1
        assert snapshot.abstained_fraction == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_summary_survives_an_abstained_only_situation(self) -> None:
        store = SituationStore(
            ttl=timedelta(minutes=30),
            dormancy_window=timedelta(minutes=10),
            correlation=EntityCorrelation(),
        )
        await store.ingest(UNSWThreatAdapter().adapt(_sensor_payload()))
        async with store._lock:
            situation = next(iter(store._situations.values()))
        summary = situation.summary(
            dormancy_window=timedelta(minutes=10), ttl=timedelta(minutes=30)
        )
        assert summary["evidence_count"] == 1
        assert summary["signal_types"] == ["unknown"]
        assert summary["max_anomaly"] == pytest.approx(0.77)
