"""FastAPI dependency injection for shared resources."""

from __future__ import annotations

from enigma_reason.services.connection_manager import ConnectionManager

# Singleton connection manager for frontend WebSocket clients
ui_manager = ConnectionManager()
