"""enigma-reason: FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from enigma_reason.api import ws_decisions, ws_signals
from enigma_reason.config import settings

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description="Agentic reasoning layer for the Enigma distributed security system",
    version="0.1.0",
)

# ── WebSocket Routes ─────────────────────────────────────────────────────────

app.include_router(ws_signals.router, tags=["signals"])
app.include_router(ws_decisions.router, tags=["decisions"])


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
