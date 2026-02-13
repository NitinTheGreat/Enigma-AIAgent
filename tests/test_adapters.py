"""Tests for Phase 3: Normalization & Signal Adapters.

Tests adapter selection, payload rejection, score normalization,
canonical Signal validation after adaptation, and registry stats.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from enigma_reason.adapters.auth import AuthAnomalyAdapter
from enigma_reason.adapters.network import NetworkAnomalyAdapter
from enigma_reason.adapters.registry import (
    AdaptationError,
    AdapterRegistry,
    NoAdapterFoundError,
)
from enigma_reason.adapters.video import VideoDetectionAdapter
from enigma_reason.domain.signal import Signal


# ── Realistic Raw Payloads ───────────────────────────────────────────────────

_NOW = datetime.now(timezone.utc).isoformat()


def _network_payload(**overrides) -> dict:
    base = {
        "source_type": "network_anomaly",
        "src_ip": "10.0.0.42",
        "dst_ip": "203.0.113.5",
        "protocol": "tcp",
        "bytes_sent": 1048576,
        "bytes_received": 256,
        "z_score": 4.2,
        "detector_id": "net-detector-01",
        "timestamp": _NOW,
    }
    base.update(overrides)
    return base


def _auth_payload(**overrides) -> dict:
    base = {
        "source_type": "auth_anomaly",
        "username": "alice",
        "failed_attempts": 15,
        "window_seconds": 60,
        "source_ip": "192.168.1.50",
        "detector_id": "auth-detector-01",
        "timestamp": _NOW,
    }
    base.update(overrides)
    return base


def _video_payload(**overrides) -> dict:
    base = {
        "source_type": "video_detection",
        "camera_id": "cam-lobby-01",
        "person_detected": True,
        "object_class": "person",
        "confidence": 0.92,
        "zone": "restricted_area",
        "detector_id": "vision-detector-01",
        "timestamp": _NOW,
    }
    base.update(overrides)
    return base


def _build_registry() -> AdapterRegistry:
    reg = AdapterRegistry()
    reg.register(NetworkAnomalyAdapter())
    reg.register(AuthAnomalyAdapter())
    reg.register(VideoDetectionAdapter())
    return reg


# ── Adapter Selection Tests ──────────────────────────────────────────────────


class TestAdapterSelection:
    def test_network_adapter_selected(self) -> None:
        reg = _build_registry()
        signal = reg.adapt(_network_payload())
        assert isinstance(signal, Signal)
        assert signal.source == "net-detector-01"

    def test_auth_adapter_selected(self) -> None:
        reg = _build_registry()
        signal = reg.adapt(_auth_payload())
        assert isinstance(signal, Signal)
        assert signal.source == "auth-detector-01"

    def test_video_adapter_selected(self) -> None:
        reg = _build_registry()
        signal = reg.adapt(_video_payload())
        assert isinstance(signal, Signal)
        assert signal.source == "vision-detector-01"

    def test_no_adapter_raises(self) -> None:
        reg = _build_registry()
        with pytest.raises(NoAdapterFoundError):
            reg.adapt({"source_type": "unknown_thing", "timestamp": _NOW})

    def test_empty_registry_raises(self) -> None:
        reg = AdapterRegistry()
        with pytest.raises(NoAdapterFoundError):
            reg.adapt(_network_payload())


# ── Payload Rejection Tests ──────────────────────────────────────────────────


class TestPayloadRejection:
    def test_network_missing_src_ip(self) -> None:
        reg = _build_registry()
        payload = _network_payload()
        del payload["src_ip"]
        with pytest.raises(AdaptationError) as exc_info:
            reg.adapt(payload)
        assert exc_info.value.adapter_name == "network_anomaly"

    def test_auth_missing_username(self) -> None:
        reg = _build_registry()
        payload = _auth_payload()
        del payload["username"]
        with pytest.raises(AdaptationError) as exc_info:
            reg.adapt(payload)
        assert exc_info.value.adapter_name == "auth_anomaly"

    def test_video_missing_camera_id(self) -> None:
        reg = _build_registry()
        payload = _video_payload()
        del payload["camera_id"]
        with pytest.raises(AdaptationError) as exc_info:
            reg.adapt(payload)
        assert exc_info.value.adapter_name == "video_detection"

    def test_network_missing_timestamp(self) -> None:
        reg = _build_registry()
        payload = _network_payload()
        del payload["timestamp"]
        with pytest.raises(AdaptationError):
            reg.adapt(payload)


# ── Score Normalization Tests ────────────────────────────────────────────────


class TestScoreNormalization:
    def test_network_z_score_normalized(self) -> None:
        reg = _build_registry()
        # z_score = 5.0 → anomaly_score = 5/10 = 0.5
        signal = reg.adapt(_network_payload(z_score=5.0))
        assert signal.anomaly_score == pytest.approx(0.5)

    def test_network_z_score_clamped_high(self) -> None:
        reg = _build_registry()
        # z_score = 15.0 → anomaly_score = min(15/10, 1.0) = 1.0
        signal = reg.adapt(_network_payload(z_score=15.0))
        assert signal.anomaly_score == pytest.approx(1.0)

    def test_network_z_score_zero(self) -> None:
        reg = _build_registry()
        signal = reg.adapt(_network_payload(z_score=0.0))
        assert signal.anomaly_score == pytest.approx(0.0)

    def test_auth_attempts_normalized(self) -> None:
        reg = _build_registry()
        # 10 failures → 10/20 = 0.5
        signal = reg.adapt(_auth_payload(failed_attempts=10))
        assert signal.anomaly_score == pytest.approx(0.5)

    def test_auth_attempts_clamped_high(self) -> None:
        reg = _build_registry()
        # 50 failures → min(50/20, 1.0) = 1.0
        signal = reg.adapt(_auth_payload(failed_attempts=50))
        assert signal.anomaly_score == pytest.approx(1.0)

    def test_video_confidence_passthrough(self) -> None:
        reg = _build_registry()
        # confidence = 0.92, person in restricted → 0.92 * 1.5 = 1.0 (clamped)
        signal = reg.adapt(_video_payload(confidence=0.92, zone="restricted_area"))
        assert 0.0 <= signal.anomaly_score <= 1.0

    def test_video_no_person_low_score(self) -> None:
        reg = _build_registry()
        signal = reg.adapt(_video_payload(person_detected=False, zone="lobby"))
        assert signal.anomaly_score == pytest.approx(0.1)


# ── Canonical Signal Validation Tests ────────────────────────────────────────


class TestCanonicalSignalValidation:
    def test_network_produces_valid_signal(self) -> None:
        adapter = NetworkAnomalyAdapter()
        signal = adapter.adapt(_network_payload())
        assert signal.signal_id is not None
        assert 0.0 <= signal.anomaly_score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.entity is not None
        assert signal.entity.identifier == "10.0.0.42"

    def test_auth_produces_valid_signal(self) -> None:
        adapter = AuthAnomalyAdapter()
        signal = adapter.adapt(_auth_payload())
        assert signal.signal_id is not None
        assert signal.entity is not None
        assert signal.entity.identifier == "alice"

    def test_video_produces_valid_signal(self) -> None:
        adapter = VideoDetectionAdapter()
        signal = adapter.adapt(_video_payload())
        assert signal.signal_id is not None
        assert signal.entity is not None
        assert signal.entity.identifier == "cam-lobby-01"

    def test_adapted_signal_is_immutable(self) -> None:
        adapter = NetworkAnomalyAdapter()
        signal = adapter.adapt(_network_payload())
        with pytest.raises(Exception):
            signal.source = "tampered"

    def test_network_features_populated(self) -> None:
        adapter = NetworkAnomalyAdapter()
        signal = adapter.adapt(_network_payload())
        assert len(signal.features) > 0
        assert any("proto:" in f for f in signal.features)

    def test_auth_features_populated(self) -> None:
        adapter = AuthAnomalyAdapter()
        signal = adapter.adapt(_auth_payload())
        assert len(signal.features) > 0
        assert any("fail_rate:" in f for f in signal.features)

    def test_video_features_populated(self) -> None:
        adapter = VideoDetectionAdapter()
        signal = adapter.adapt(_video_payload())
        assert len(signal.features) > 0
        assert "person_detected" in signal.features


# ── Registry Stats Tests ─────────────────────────────────────────────────────


class TestRegistryStats:
    def test_stats_track_accepted(self) -> None:
        reg = _build_registry()
        reg.adapt(_network_payload())
        reg.adapt(_network_payload())
        stats = {s["adapter_name"]: s for s in reg.stats}
        assert stats["network_anomaly"]["accepted_count"] == 2

    def test_stats_track_rejected(self) -> None:
        reg = _build_registry()
        bad = _network_payload()
        del bad["src_ip"]
        with pytest.raises(AdaptationError):
            reg.adapt(bad)
        stats = {s["adapter_name"]: s for s in reg.stats}
        assert stats["network_anomaly"]["rejected_count"] == 1

    def test_total_accepted(self) -> None:
        reg = _build_registry()
        reg.adapt(_network_payload())
        reg.adapt(_auth_payload())
        reg.adapt(_video_payload())
        assert reg.total_accepted == 3

    def test_total_rejected(self) -> None:
        reg = _build_registry()
        bad = _auth_payload()
        del bad["username"]
        with pytest.raises(AdaptationError):
            reg.adapt(bad)
        assert reg.total_rejected == 1

    def test_adapter_names(self) -> None:
        reg = _build_registry()
        assert reg.adapter_names == ["network_anomaly", "auth_anomaly", "video_detection"]

    def test_payload_not_mutated(self) -> None:
        """Adapters must not mutate the input dict."""
        reg = _build_registry()
        payload = _network_payload()
        original_keys = set(payload.keys())
        original_values = dict(payload)
        reg.adapt(payload)
        assert set(payload.keys()) == original_keys
        assert payload == original_values


# ── can_handle Tests ─────────────────────────────────────────────────────────


class TestCanHandle:
    def test_network_can_handle_yes(self) -> None:
        assert NetworkAnomalyAdapter().can_handle({"source_type": "network_anomaly"})

    def test_network_can_handle_no(self) -> None:
        assert not NetworkAnomalyAdapter().can_handle({"source_type": "other"})

    def test_auth_can_handle_yes(self) -> None:
        assert AuthAnomalyAdapter().can_handle({"source_type": "auth_anomaly"})

    def test_video_can_handle_yes(self) -> None:
        assert VideoDetectionAdapter().can_handle({"source_type": "video_detection"})

    def test_missing_source_type(self) -> None:
        assert not NetworkAnomalyAdapter().can_handle({"something": "else"})
