"""VideoDetectionAdapter — translates raw video detection payloads.

Expected raw format:
{
    "source_type": "video_detection",
    "camera_id": "cam-lobby-01",
    "person_detected": true,
    "object_class": "person",
    "confidence": 0.92,
    "zone": "restricted_area",
    "detector_id": "vision-detector-01",
    "timestamp": "2026-02-13T14:00:00Z"
}
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from enigma_reason.adapters.base import SignalAdapter
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import Signal


class VideoDetectionAdapter(SignalAdapter):
    """Maps video detection payloads to canonical Signals."""

    @property
    def source_name(self) -> str:
        return "video_detection"

    def can_handle(self, raw: dict[str, Any]) -> bool:
        return raw.get("source_type") == "video_detection"

    def adapt(self, raw: dict[str, Any]) -> Signal:
        # ── Extract required fields ──────────────────────────────────────
        camera_id = raw.get("camera_id")
        if not camera_id:
            raise ValueError("video_detection payload missing 'camera_id'")

        timestamp = raw.get("timestamp")
        if not timestamp:
            raise ValueError("video_detection payload missing 'timestamp'")

        # ── Normalise confidence (already 0–1 from vision model) ─────────
        raw_confidence = raw.get("confidence", 0.5)
        confidence = max(0.0, min(float(raw_confidence), 1.0))

        # ── Anomaly score: based on zone + detection ─────────────────────
        zone = raw.get("zone", "")
        person_detected = raw.get("person_detected", False)
        anomaly_score = confidence if person_detected else 0.1
        if zone in ("restricted_area", "perimeter", "server_room"):
            anomaly_score = min(anomaly_score * 1.5, 1.0)

        # ── Build features ───────────────────────────────────────────────
        features: list[str] = []
        if raw.get("object_class"):
            features.append(f"class:{raw['object_class']}")
        if zone:
            features.append(f"zone:{zone}")
        if person_detected:
            features.append("person_detected")

        # ── Determine signal type ────────────────────────────────────────
        signal_type = SignalType.RECONNAISSANCE
        if zone in ("restricted_area", "server_room"):
            signal_type = SignalType.POLICY_VIOLATION

        return Signal.model_validate({
            "signal_id": str(uuid4()),
            "timestamp": timestamp,
            "signal_type": signal_type.value,
            "entity": {"kind": EntityKind.DEVICE.value, "identifier": camera_id},
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "features": features,
            "source": raw.get("detector_id", "vision-unknown"),
        })
