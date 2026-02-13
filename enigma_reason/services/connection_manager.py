"""Manages active WebSocket connections for broadcasting decisions to UI clients."""

from __future__ import annotations

from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Thread-safe manager for frontend WebSocket connections."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.remove(websocket)

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        """Send a JSON payload to every connected UI client."""
        for ws in self._connections:
            await ws.send_json(data)
