"""Tests for the canonical Signal model."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import EntityRef, Signal


def _valid_signal(**overrides) -> dict:
    """Return a valid signal dict, with optional overrides."""
    base = {
        "signal_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_type": "intrusion",
        "entity": {"kind": "user", "identifier": "alice"},
        "anomaly_score": 0.85,
        "confidence": 0.9,
        "features": ["login_burst", "geo_anomaly"],
        "source": "detector-alpha",
    }
    base.update(overrides)
    return base


class TestSignalValidation:
    def test_valid_signal_parses(self) -> None:
        signal = Signal.model_validate(_valid_signal())
        assert signal.signal_type == SignalType.INTRUSION
        assert signal.anomaly_score == 0.85

    def test_entity_ref_is_optional(self) -> None:
        signal = Signal.model_validate(_valid_signal(entity=None))
        assert signal.entity is None

    def test_anomaly_score_out_of_range_rejected(self) -> None:
        with pytest.raises(Exception):
            Signal.model_validate(_valid_signal(anomaly_score=1.5))

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(Exception):
            Signal.model_validate(_valid_signal(confidence=-0.1))

    def test_future_timestamp_accepted(self) -> None:
        """Future timestamps are accepted (ML may replay historical data)."""
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        signal = Signal.model_validate(_valid_signal(timestamp=future))
        assert signal.timestamp.tzinfo is not None

    def test_naive_timestamp_gets_utc(self) -> None:
        """Naive timestamps get UTC auto-attached."""
        naive = datetime.now().isoformat()
        signal = Signal.model_validate(_valid_signal(timestamp=naive))
        assert signal.timestamp.tzinfo is not None

    def test_invalid_signal_type_rejected(self) -> None:
        with pytest.raises(Exception):
            Signal.model_validate(_valid_signal(signal_type="made_up_type"))

    def test_long_feature_tag_rejected(self) -> None:
        with pytest.raises(Exception):
            Signal.model_validate(_valid_signal(features=["x" * 100]))

    def test_signal_is_immutable(self) -> None:
        signal = Signal.model_validate(_valid_signal())
        with pytest.raises(Exception):
            signal.source = "changed"


class TestEntityRef:
    def test_str_format(self) -> None:
        ref = EntityRef(kind=EntityKind.DEVICE, identifier="laptop-42")
        assert str(ref) == "device:laptop-42"
