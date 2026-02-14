"""Dashboard WebSocket — pushes live analysis to connected frontends.

Architecture:
    ML  →  /ws/signal       →  AI Layer ingests signal
                                    ↓
                               runs reasoning + builds explanation
                                    ↓
    FE  ←  /ws/dashboard    ←  broadcasts analysis to all connected clients

The DashboardManager is a singleton that tracks connected clients
and broadcasts analysis results whenever a situation is updated.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from enigma_reason.core.reasoning_engine import ReasoningEngine
from enigma_reason.domain.explanation import filter_explanation_for_role, ExplanationRole
from enigma_reason.domain.situation import Situation
from enigma_reason.explain.builder import build_explanation
from enigma_reason.explain.formatter import ExplanationFormatter
from enigma_reason.graph.runner import run_reasoning

logger = logging.getLogger(__name__)


class DashboardManager:
    """Tracks connected frontend WebSocket clients and broadcasts analyses."""

    def __init__(
        self,
        reasoning_engine: ReasoningEngine,
        burst_factor: float = 3.0,
        burst_recent_count: int = 3,
        quiet_window: timedelta = timedelta(minutes=5),
    ) -> None:
        self._reasoning_engine = reasoning_engine
        self._burst_factor = burst_factor
        self._burst_recent_count = burst_recent_count
        self._quiet_window = quiet_window
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ── Client management ────────────────────────────────────────────

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info("Dashboard client connected (%d total)", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info("Dashboard client disconnected (%d remaining)", len(self._clients))

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── Analysis + broadcast ─────────────────────────────────────────

    async def on_situation_updated(self, situation: Situation) -> None:
        """Called after signal ingestion. Runs reasoning and pushes to FE.

        Runs in a background task so it doesn't block signal ingestion.
        """
        if not self._clients:
            return  # No frontends connected, skip

        try:
            payload = await self._build_analysis(situation)
            await self._broadcast(payload)
        except Exception as exc:
            logger.error("Dashboard analysis/broadcast failed: %s", exc, exc_info=True)

    async def _build_analysis(self, situation: Situation) -> dict[str, Any]:
        """Run the full analysis pipeline for a situation."""

        # Phase 2: Temporal snapshot
        temporal = situation.temporal_snapshot(
            burst_factor=self._burst_factor,
            recent_count=self._burst_recent_count,
            quiet_window=self._quiet_window,
        )

        # Phase 4: Deterministic reasoning
        reasoning = self._reasoning_engine.evaluate(situation)

        # Phase 5: LangGraph reasoning (runs Gemini — offloaded to thread
        # pool so we don't block the event loop and starve other WebSockets)
        try:
            final_state = await asyncio.to_thread(
                run_reasoning, situation, temporal, reasoning
            )
        except Exception as exc:
            logger.warning("LangGraph reasoning failed, using fallback: %s", exc)
            final_state = {
                "situation_id": str(situation.situation_id),
                "hypotheses": [],
                "convergence_score": 0.0,
                "belief_stability_score": 0.0,
                "iteration_count": 0,
                "undecided_iterations": 0,
                "last_confidence_shift": 0.0,
            }

        # Phase 6: Build explanation
        explanation = build_explanation(final_state, reasoning, temporal)
        human_text = ExplanationFormatter.format_plain(explanation)

        return {
            "type": "situation_analysis",

            # Situation facts
            "situation": situation.summary(),

            # Temporal
            "temporal": temporal.model_dump(),

            # Reasoning
            "reasoning": reasoning.model_dump(),

            # LangGraph
            "langgraph": {
                "hypotheses": final_state.get("hypotheses", []),
                "convergence_score": final_state.get("convergence_score", 0.0),
                "iterations": final_state.get("iteration_count", 0),
                "belief_stability": final_state.get("belief_stability_score", 0.0),
                "undecided_iterations": final_state.get("undecided_iterations", 0),
            },

            # Explanation
            "explanation": {
                "undecided": explanation.undecided,
                "dominant_hypothesis_id": explanation.dominant_hypothesis_id,
                "dominant_confidence": explanation.dominant_confidence,
                "convergence_score": explanation.convergence_score,
                "sections": [
                    {
                        "type": s.section_type.value,
                        "title": s.title,
                        "bullets": s.bullet_points,
                        "contribution_score": s.contribution_score,
                        "contribution_direction": (
                            s.contribution_direction.value if s.contribution_direction else None
                        ),
                        "counterfactuals": [
                            {
                                "missing_condition": cf.missing_condition,
                                "expected_effect": cf.expected_effect,
                                "confidence_delta": cf.confidence_delta_estimate,
                            }
                            for cf in s.counterfactuals
                        ] if s.counterfactuals else None,
                    }
                    for s in explanation.explanation_sections
                ],
                "temporal_evolution": (
                    {
                        "confidence_trend": explanation.temporal_evolution.confidence_trend,
                        "velocity": explanation.temporal_evolution.belief_velocity_summary,
                        "stability": explanation.temporal_evolution.stability_label,
                        "undecided_duration": explanation.temporal_evolution.undecided_duration,
                    }
                    if explanation.temporal_evolution else None
                ),
            },

            # Human-readable
            "human_readable": human_text,
        }

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        """Send payload to all connected dashboard clients."""
        message = json.dumps(payload, default=str)
        dead: set[WebSocket] = set()

        async with self._lock:
            clients = set(self._clients)

        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)

        if dead:
            async with self._lock:
                self._clients -= dead
            logger.info("Removed %d dead dashboard client(s)", len(dead))


# ── WebSocket endpoint ───────────────────────────────────────────────────


def create_dashboard_router(manager: DashboardManager) -> APIRouter:
    """Factory that creates the dashboard WebSocket endpoint."""

    router = APIRouter()

    @router.websocket("/ws/dashboard")
    async def dashboard_ws(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            # Keep the connection alive — FE just listens
            while True:
                # Accept any message from FE (heartbeats, pings)
                data = await websocket.receive_text()
                # If FE sends "ping", reply "pong"
                if data.strip().lower() == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            await manager.disconnect(websocket)

    return router
