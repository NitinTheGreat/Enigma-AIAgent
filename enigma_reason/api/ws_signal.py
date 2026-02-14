"""WebSocket endpoint for signal ingestion.

Path: /ws/signal

Accepts JSON matching the Signal schema, validates it at the boundary,
routes it into the SituationStore, and returns a minimal acknowledgement.
If strict validation fails, falls back to the adapter registry.
If a DashboardManager is attached, triggers background analysis + push.

No decisions.  No alerts.  No LLM calls on this path.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from enigma_reason.adapters.registry import AdapterRegistry
from enigma_reason.domain.signal import Signal
from enigma_reason.store.situation_store import SituationStore

router = APIRouter()
logger = logging.getLogger(__name__)


def create_signal_router(
    store: SituationStore,
    dashboard_manager=None,
    adapter_registry: AdapterRegistry | None = None,
) -> APIRouter:
    """Factory that wires the signal endpoint to a concrete SituationStore.

    Args:
        store: The SituationStore to ingest signals into.
        dashboard_manager: Optional DashboardManager to push analysis to FE.
        adapter_registry: Optional AdapterRegistry for fallback conversion.
    """

    @router.websocket("/ws/signal")
    async def ingest_signal(websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("Signal source connected")

        try:
            while True:
                raw = await websocket.receive_json()

                # ── Validate at the boundary ─────────────────────────────
                signal = None
                try:
                    signal = Signal.model_validate(raw)
                except ValidationError:
                    # Strict validation failed — try adapter fallback
                    if adapter_registry is not None:
                        try:
                            signal = adapter_registry.adapt(raw)
                            logger.info("Adapted raw signal via registry")
                        except Exception as adapt_exc:
                            logger.debug("Adapter fallback failed: %s", adapt_exc)

                if signal is None:
                    await websocket.send_json({
                        "status": "error",
                        "detail": "Signal validation failed and no adapter could handle the format",
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

                # ── Push to dashboard (background, non-blocking) ─────────
                if dashboard_manager is not None:
                    asyncio.create_task(
                        dashboard_manager.on_situation_updated(situation)
                    )

        except WebSocketDisconnect:
            logger.info("Signal source disconnected")

    return router

