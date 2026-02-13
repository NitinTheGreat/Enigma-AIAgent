"""WebSocket endpoint: receives structured signals from the ML service."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from enigma_reason.models.signal import IncomingSignal

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/signals")
async def receive_signals(websocket: WebSocket) -> None:
    """Ingest signals pushed by the external ML inference service."""
    await websocket.accept()
    logger.info("ML signal source connected")

    try:
        while True:
            raw = await websocket.receive_json()
            signal = IncomingSignal.model_validate(raw)
            logger.debug("Signal received: %s [%s]", signal.signal_id, signal.signal_type)

            # TODO: feed signal into the reasoning graph
            # result = await reasoning_graph.ainvoke({"raw_signal": raw, ...})

            await websocket.send_json({"status": "ack", "signal_id": str(signal.signal_id)})

    except WebSocketDisconnect:
        logger.info("ML signal source disconnected")
