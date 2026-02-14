"""UNSWThreatAdapter — translates UNSW threat detector ML output to canonical Signals.

Expected raw format (full ML output):
{
    "input_that_i_gave_to_the_model": { ... },
    "raw_output_from_model": [...],
    "output_from_model": "⚠️ THREAT DETECTED: backdoor (Confidence: 0.42)",
    "inputs_for_xai_model": {
        "signal_id": "uuid",
        "timestamp": "2015-02-18T07:26:36Z",
        "signal_type": "backdoor",
        "entity": {"device": "...", "user": "...", "location": "..."},
        "anomaly_score": 0.42,
        "confidence": 0.42,
        "features": ["dur", "sbytes", ...],
        "source": "unsw-threat-detector"
    }
}

Also handles the inner 'inputs_for_xai_model' format directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from enigma_reason.adapters.base import SignalAdapter
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import EntityRef, Signal


# Map ML signal_type strings to our canonical SignalType enum
_SIGNAL_TYPE_MAP: dict[str, SignalType] = {
    "backdoor": SignalType.BACKDOOR,
    "dos": SignalType.DOS,
    "exploit": SignalType.EXPLOIT,
    "exploits": SignalType.EXPLOIT,
    "fuzzers": SignalType.FUZZERS,
    "shellcode": SignalType.SHELLCODE,
    "worms": SignalType.WORMS,
    "generic": SignalType.GENERIC,
    "analysis": SignalType.ANALYSIS,
    "normal": SignalType.NORMAL,
    "reconnaissance": SignalType.RECONNAISSANCE,
    "intrusion": SignalType.INTRUSION,
    "anomalous_access": SignalType.ANOMALOUS_ACCESS,
    "data_exfiltration": SignalType.DATA_EXFILTRATION,
    "privilege_escalation": SignalType.PRIVILEGE_ESCALATION,
    "lateral_movement": SignalType.LATERAL_MOVEMENT,
    "policy_violation": SignalType.POLICY_VIOLATION,
}


class UNSWThreatAdapter(SignalAdapter):
    """Maps UNSW threat detector ML output to canonical Signals."""

    @property
    def source_name(self) -> str:
        return "unsw_threat"

    def can_handle(self, raw: dict[str, Any]) -> bool:
        """Accept either the full ML output or the inner inputs_for_xai_model."""
        # Full ML output format
        if "inputs_for_xai_model" in raw:
            return True
        # Direct inner format (has signal_type + source containing "unsw" or
        # has entity with device/user/location keys)
        if "signal_type" in raw and "source" in raw:
            entity = raw.get("entity", {})
            if isinstance(entity, dict) and any(
                k in entity for k in ("device", "user", "location")
            ):
                return True
        return False

    def adapt(self, raw: dict[str, Any]) -> Signal:
        # ── Extract the xai payload (handle both wrapper and direct) ─────
        xai = raw.get("inputs_for_xai_model", raw)

        # ── Signal ID ────────────────────────────────────────────────────
        signal_id = xai.get("signal_id", str(uuid4()))

        # ── Timestamp ────────────────────────────────────────────────────
        ts_raw = xai.get("timestamp")
        if ts_raw:
            if isinstance(ts_raw, str):
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            else:
                ts = ts_raw
        else:
            ts = datetime.now(timezone.utc)

        # ── Signal Type ──────────────────────────────────────────────────
        raw_type = xai.get("signal_type", "unknown").lower().strip()
        signal_type = _SIGNAL_TYPE_MAP.get(raw_type, SignalType.UNKNOWN)

        # ── Entity ───────────────────────────────────────────────────────
        # ML sends: {"device": "x", "user": "y", "location": "z"}
        # We need: EntityRef(kind=..., identifier=...)
        entity_raw = xai.get("entity", {})
        entity = None
        if isinstance(entity_raw, dict):
            if "kind" in entity_raw and "identifier" in entity_raw:
                # Already in our format
                entity = EntityRef(
                    kind=EntityKind(entity_raw["kind"]),
                    identifier=entity_raw["identifier"],
                )
            else:
                # ML format: pick the first available identifier
                if entity_raw.get("device"):
                    entity = EntityRef(
                        kind=EntityKind.DEVICE,
                        identifier=entity_raw["device"],
                    )
                elif entity_raw.get("user"):
                    entity = EntityRef(
                        kind=EntityKind.USER,
                        identifier=entity_raw["user"],
                    )
                elif entity_raw.get("location"):
                    entity = EntityRef(
                        kind=EntityKind.LOCATION,
                        identifier=entity_raw["location"],
                    )

        # ── Scores ───────────────────────────────────────────────────────
        anomaly_score = float(xai.get("anomaly_score", 0.5))
        anomaly_score = max(0.0, min(1.0, anomaly_score))

        confidence = float(xai.get("confidence", anomaly_score))
        confidence = max(0.0, min(1.0, confidence))

        # ── Features ────────────────────────────────────────────────────
        features = xai.get("features", [])
        if isinstance(features, list):
            features = [str(f)[:64] for f in features[:50]]
        else:
            features = []

        # ── Source ───────────────────────────────────────────────────────
        source = xai.get("source", "unsw-threat-detector")

        return Signal.model_validate({
            "signal_id": str(signal_id),
            "timestamp": ts.isoformat(),
            "signal_type": signal_type.value,
            "entity": {"kind": entity.kind.value, "identifier": entity.identifier} if entity else None,
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "features": features,
            "source": source,
        })
