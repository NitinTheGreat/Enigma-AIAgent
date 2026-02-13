"""WebSocket endpoint for signal ingestion.

Path: /ws/signal

Accepts JSON matching the Signal schema, validates it at the boundary,
routes it into the SituationStore, and returns a minimal acknowledgement.

No decisions.  No alerts.  No LLM calls.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from enigma_reason.domain.signal import Signal
from enigma_reason.store.situation_store import SituationStore

router = APIRouter()
logger = logging.getLogger(__name__)


def create_signal_router(store: SituationStore) -> APIRouter:
    """Factory that wires the signal endpoint to a concrete SituationStore.

    This avoids module-level singletons and makes the endpoint testable.
    """

    @router.websocket("/ws/signal")
    async def ingest_signal(websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("Signal source connected")

        try:
            while True:
                raw = await websocket.receive_json()

                # ── Validate at the boundary ─────────────────────────────
                try:
                    signal = Signal.model_validate(raw)
                except ValidationError as exc:
                    await websocket.send_json({
                        "status": "error",
                        "detail": exc.errors(),
                    })
                    continue

                # ── Route into store ─────────────────────────────────────
                situation = await store.ingest(signal)

                # ── Acknowledge ──────────────────────────────────────────
                await websocket.send_json({
                    "status": "accepted",
                    "situation_id": str(situation.situation_id),
                    "evidence_count": situation.evidence_count,
                })

        except WebSocketDisconnect:
            logger.info("Signal source disconnected")

    return router
