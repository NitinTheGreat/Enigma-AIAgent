"""enigma-reason — Situation Memory & Signal Grounding + Temporal Awareness.

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
    burst_factor=settings.burst_factor,
    burst_recent_count=settings.burst_recent_count,
    quiet_window=timedelta(minutes=settings.quiet_window_minutes),
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description="Situation Memory & Signal Grounding + Temporal Awareness",
    version="0.2.0",
)

# ── Routes ───────────────────────────────────────────────────────────────────

app.include_router(create_signal_router(store))


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    ts = await store.temporal_summary()
    return {
        "status": "ok",
        "phase": 2,
        "active_situations": ts.active_situations,
        "dormant_situations": ts.dormant_situations,
        "bursting_situations": ts.bursting_situations,
        "quiet_situations": ts.quiet_situations,
        "max_event_rate": ts.max_event_rate,
    }
