"""enigma-reason — Phase 1: Situation Memory & Signal Grounding.

This is the application entry point.  It wires the SituationStore to the
WebSocket transport and starts the FastAPI server.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import FastAPI

from enigma_reason.api.ws_signal import create_signal_router
from enigma_reason.config import settings
from enigma_reason.store.situation_store import SituationStore

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ── State ────────────────────────────────────────────────────────────────────

store = SituationStore(
    ttl=timedelta(minutes=settings.situation_ttl_minutes),
    dormancy_window=timedelta(minutes=settings.situation_dormancy_minutes),
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description="Phase 1 — Situation Memory & Signal Grounding",
    version="0.1.0",
)

# ── Routes ───────────────────────────────────────────────────────────────────

app.include_router(create_signal_router(store))


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "phase": 1,
        "active_situations": await store.active_count(),
        "dormant_situations": await store.dormant_count(),
    }
