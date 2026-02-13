"""WebSocket endpoint for raw signal ingestion via adapter layer.

Path: /ws/raw-signal

Accepts raw JSON payloads from heterogeneous upstream sources, routes
them through the AdapterRegistry to produce canonical Signals, then
forwards those Signals into the SituationStore.

This is additive — /ws/signal remains unchanged.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from enigma_reason.adapters.registry import (
    AdaptationError,
    AdapterRegistry,
    NoAdapterFoundError,
)
from enigma_reason.store.situation_store import SituationStore

logger = logging.getLogger(__name__)


def create_raw_signal_router(
    store: SituationStore,
    registry: AdapterRegistry,
) -> APIRouter:
    """Factory that wires the raw-signal endpoint to store + registry."""

    router = APIRouter()

    @router.websocket("/ws/raw-signal")
    async def ingest_raw_signal(websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("Raw signal source connected")

        try:
            while True:
                raw = await websocket.receive_json()

                # ── Route through adapter registry ───────────────────────
                try:
                    signal = registry.adapt(raw)
                except NoAdapterFoundError as exc:
                    await websocket.send_json({
                        "status": "error",
                        "reason": "no_adapter",
                        "detail": str(exc),
                    })
                    continue
                except AdaptationError as exc:
                    await websocket.send_json({
                        "status": "error",
                        "reason": "adaptation_failed",
                        "adapter": exc.adapter_name,
                        "detail": exc.reason,
                    })
                    continue

                # ── Forward canonical Signal into store ──────────────────
                situation = await store.ingest(signal)

                # ── Acknowledge ──────────────────────────────────────────
                await websocket.send_json({
                    "status": "accepted",
                    "adapter": signal.source,
                    "situation_id": str(situation.situation_id),
                    "evidence_count": situation.evidence_count,
                })

        except WebSocketDisconnect:
            logger.info("Raw signal source disconnected")

    return router
