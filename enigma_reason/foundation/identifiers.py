"""Deterministic ID generation for domain objects."""

from __future__ import annotations

from uuid import uuid4, UUID


def new_id() -> UUID:
    """Generate a new random UUID v4 for domain objects."""
    return uuid4()
