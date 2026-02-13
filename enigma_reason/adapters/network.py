"""NetworkAnomalyAdapter — translates raw network anomaly payloads.

Expected raw format:
{
    "source_type": "network_anomaly",
    "src_ip": "10.0.0.42",
    "dst_ip": "203.0.113.5",
    "protocol": "tcp",
    "bytes_sent": 1048576,
    "bytes_received": 256,
    "z_score": 4.2,
    "detector_id": "net-detector-01",
    "timestamp": "2026-02-13T14:00:00Z"
}
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from enigma_reason.adapters.base import SignalAdapter
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import EntityRef, Signal


class NetworkAnomalyAdapter(SignalAdapter):
    """Maps network anomaly payloads to canonical Signals."""

    @property
    def source_name(self) -> str:
        return "network_anomaly"

    def can_handle(self, raw: dict[str, Any]) -> bool:
        return raw.get("source_type") == "network_anomaly"

    def adapt(self, raw: dict[str, Any]) -> Signal:
        # ── Extract required fields ──────────────────────────────────────
        src_ip = raw.get("src_ip")
        if not src_ip:
            raise ValueError("network_anomaly payload missing 'src_ip'")

        z_score = raw.get("z_score")
        if z_score is None:
            raise ValueError("network_anomaly payload missing 'z_score'")

        timestamp = raw.get("timestamp")
        if not timestamp:
            raise ValueError("network_anomaly payload missing 'timestamp'")

        # ── Normalise z_score → anomaly_score (0–1) ─────────────────────
        # z_score ∈ [0, ∞) → clamp to [0, 1] via sigmoid-like mapping:
        # score = min(z / 10, 1.0) — simple, transparent, no ML
        anomaly_score = min(abs(float(z_score)) / 10.0, 1.0)

        # ── Build features ───────────────────────────────────────────────
        features: list[str] = []
        if raw.get("protocol"):
            features.append(f"proto:{raw['protocol']}")
        bytes_sent = raw.get("bytes_sent", 0)
        bytes_recv = raw.get("bytes_received", 0)
        if bytes_sent > 0 and bytes_recv > 0:
            ratio = bytes_sent / bytes_recv
            if ratio > 10:
                features.append("high_send_ratio")
        if abs(float(z_score)) > 3:
            features.append("high_z_score")

        # ── Determine signal type ────────────────────────────────────────
        signal_type = SignalType.INTRUSION
        if bytes_sent > 500_000:
            signal_type = SignalType.DATA_EXFILTRATION

        return Signal.model_validate({
            "signal_id": str(uuid4()),
            "timestamp": timestamp,
            "signal_type": signal_type.value,
            "entity": {"kind": EntityKind.DEVICE.value, "identifier": src_ip},
            "anomaly_score": anomaly_score,
            "confidence": min(abs(float(z_score)) / 5.0, 1.0),
            "features": features,
            "source": raw.get("detector_id", "network-unknown"),
        })
