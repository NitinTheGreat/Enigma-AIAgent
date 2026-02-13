"""AuthAnomalyAdapter — translates raw authentication anomaly payloads.

Expected raw format:
{
    "source_type": "auth_anomaly",
    "username": "alice",
    "failed_attempts": 15,
    "window_seconds": 60,
    "source_ip": "192.168.1.50",
    "detector_id": "auth-detector-01",
    "timestamp": "2026-02-13T14:00:00Z"
}
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from enigma_reason.adapters.base import SignalAdapter
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import EntityRef, Signal


class AuthAnomalyAdapter(SignalAdapter):
    """Maps authentication anomaly payloads to canonical Signals."""

    @property
    def source_name(self) -> str:
        return "auth_anomaly"

    def can_handle(self, raw: dict[str, Any]) -> bool:
        return raw.get("source_type") == "auth_anomaly"

    def adapt(self, raw: dict[str, Any]) -> Signal:
        # ── Extract required fields ──────────────────────────────────────
        username = raw.get("username")
        if not username:
            raise ValueError("auth_anomaly payload missing 'username'")

        failed_attempts = raw.get("failed_attempts")
        if failed_attempts is None:
            raise ValueError("auth_anomaly payload missing 'failed_attempts'")

        window_seconds = raw.get("window_seconds")
        if window_seconds is None:
            raise ValueError("auth_anomaly payload missing 'window_seconds'")

        timestamp = raw.get("timestamp")
        if not timestamp:
            raise ValueError("auth_anomaly payload missing 'timestamp'")

        # ── Normalise failed_attempts → anomaly_score (0–1) ─────────────
        # score = min(attempts / 20, 1.0) — 20+ failures within window = max
        anomaly_score = min(int(failed_attempts) / 20.0, 1.0)

        # ── Build features ───────────────────────────────────────────────
        features: list[str] = []
        rate = int(failed_attempts) / max(int(window_seconds), 1)
        features.append(f"fail_rate:{rate:.2f}/s")
        if int(failed_attempts) > 10:
            features.append("brute_force_pattern")
        if raw.get("source_ip"):
            features.append(f"src:{raw['source_ip']}")

        # ── Determine signal type ────────────────────────────────────────
        signal_type = SignalType.ANOMALOUS_ACCESS
        if int(failed_attempts) > 15:
            signal_type = SignalType.INTRUSION

        # ── Confidence: higher with more data ────────────────────────────
        confidence = min(int(failed_attempts) / 10.0, 1.0)

        return Signal.model_validate({
            "signal_id": str(uuid4()),
            "timestamp": timestamp,
            "signal_type": signal_type.value,
            "entity": {"kind": EntityKind.USER.value, "identifier": username},
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "features": features,
            "source": raw.get("detector_id", "auth-unknown"),
        })
