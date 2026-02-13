"""Controlled enumerations for the enigma-reason domain.

Every categorical field in the domain MUST reference an enum defined here.
Free-form strings are not acceptable for classification fields.
"""

from __future__ import annotations

from enum import Enum


class SignalType(str, Enum):
    """Classification labels that the upstream ML service may emit."""

    INTRUSION = "intrusion"
    ANOMALOUS_ACCESS = "anomalous_access"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    POLICY_VIOLATION = "policy_violation"
    RECONNAISSANCE = "reconnaissance"
    UNKNOWN = "unknown"


class EntityKind(str, Enum):
    """The kind of entity a signal is about."""

    USER = "user"
    DEVICE = "device"
    LOCATION = "location"
