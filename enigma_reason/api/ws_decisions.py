"""WebSocket endpoint: streams decisions and explanations to the frontend UI."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from enigma_reason.api.dependencies import ui_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/decisions")
async def stream_decisions(websocket: WebSocket) -> None:
    """Frontend clients connect here to receive live decision updates."""
    await ui_manager.connect(websocket)
    logger.info("UI client connected — total: %d", ui_manager.active_count)

    try:
        while True:
            # Keep the connection alive; decisions are pushed server-side
            await websocket.receive_text()

    except WebSocketDisconnect:
        ui_manager.disconnect(websocket)
        logger.info("UI client disconnected — total: %d", ui_manager.active_count)
