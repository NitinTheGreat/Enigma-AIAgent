"""Timezone-aware clock utilities.

All timestamps in enigma-reason MUST be UTC-aware.  This module is the
single source of "now" so tests can monkey-patch it trivially.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)
