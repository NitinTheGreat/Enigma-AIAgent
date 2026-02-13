"""REST endpoint for situation analysis — triggers reasoning + builds explanation.

Path: GET /api/situation/{id}/analyze

This wires together:
1. SituationStore (existing situation)
2. ReasoningEngine (Phase 4 deterministic snapshot)
3. run_reasoning() (Phase 5 LangGraph + Gemini)
4. build_explanation() (Phase 6 structured explanation)
5. ExplanationFormatter.format_plain() (human-readable text)

Returns a complete analysis: situation facts, reasoning state,
hypotheses, explanation sections, counterfactuals, temporal evolution,
and a human-readable narrative — all in one JSON response.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException

from enigma_reason.core.reasoning_engine import ReasoningEngine
from enigma_reason.domain.explanation import ExplanationRole, filter_explanation_for_role
from enigma_reason.explain.builder import build_explanation
from enigma_reason.explain.formatter import ExplanationFormatter
from enigma_reason.graph.runner import run_reasoning
from enigma_reason.store.situation_store import SituationStore

logger = logging.getLogger(__name__)


def create_analyze_router(
    store: SituationStore,
    reasoning_engine: ReasoningEngine,
    burst_factor: float = 3.0,
    burst_recent_count: int = 3,
    quiet_window: timedelta = timedelta(minutes=5),
) -> APIRouter:
    """Factory that wires the analyze endpoint to store + reasoning engine."""

    router = APIRouter(prefix="/api", tags=["analysis"])

    @router.get("/situations")
    async def list_situations() -> dict[str, Any]:
        """List all active situations with summaries."""
        situations = []
        async with store._lock:
            for sit in store._situations.values():
                situations.append(sit.summary())
        return {"situations": situations, "count": len(situations)}

    @router.get("/situation/{situation_id}/analyze")
    async def analyze_situation(
        situation_id: UUID,
        role: str = "ANALYST",
    ) -> dict[str, Any]:
        """Full analysis: reasoning + explanation for a situation.

        Query params:
            role: ANALYST (full), MANAGER (summary), AUDITOR (deterministic)
        """
        # ── Fetch situation ───────────────────────────────────────
        situation = await store.get(situation_id)
        if situation is None:
            raise HTTPException(status_code=404, detail=f"Situation {situation_id} not found")

        if situation.evidence_count == 0:
            raise HTTPException(status_code=400, detail="Situation has no evidence")

        # ── Phase 2: Temporal snapshot ────────────────────────────
        temporal = situation.temporal_snapshot(
            burst_factor=burst_factor,
            recent_count=burst_recent_count,
            quiet_window=quiet_window,
        )

        # ── Phase 4: Deterministic reasoning ──────────────────────
        reasoning = reasoning_engine.evaluate(situation)

        # ── Phase 5: LangGraph reasoning ──────────────────────────
        logger.info("Running LangGraph reasoning for situation %s", situation_id)
        try:
            final_state = run_reasoning(situation, temporal, reasoning)
        except Exception as exc:
            logger.error("LangGraph reasoning failed: %s", exc)
            # Fallback: build explanation from Phase 4 alone
            final_state = {
                "situation_id": str(situation_id),
                "hypotheses": [],
                "convergence_score": 0.0,
                "belief_stability_score": 0.0,
                "iteration_count": 0,
                "undecided_iterations": 0,
                "last_confidence_shift": 0.0,
            }

        # ── Phase 6: Build explanation ────────────────────────────
        explanation = build_explanation(final_state, reasoning, temporal)

        # ── Role filter ───────────────────────────────────────────
        role_enum = ExplanationRole.ANALYST
        try:
            role_enum = ExplanationRole(role.upper())
        except ValueError:
            pass
        filtered = filter_explanation_for_role(explanation, role_enum)

        # ── Format human-readable text ────────────────────────────
        human_text = ExplanationFormatter.format_plain(filtered)

        # ── Build response ────────────────────────────────────────
        return {
            # Situation facts
            "situation": situation.summary(),

            # Temporal awareness
            "temporal": temporal.model_dump(),

            # Deterministic reasoning (Phase 4)
            "reasoning": reasoning.model_dump(),

            # LangGraph state (Phase 5)
            "langgraph": {
                "hypotheses": final_state.get("hypotheses", []),
                "convergence_score": final_state.get("convergence_score", 0.0),
                "iterations": final_state.get("iteration_count", 0),
                "belief_stability": final_state.get("belief_stability_score", 0.0),
                "undecided_iterations": final_state.get("undecided_iterations", 0),
            },

            # Explanation (Phase 6/6.1)
            "explanation": {
                "undecided": filtered.undecided,
                "dominant_hypothesis_id": filtered.dominant_hypothesis_id,
                "dominant_confidence": filtered.dominant_confidence,
                "convergence_score": filtered.convergence_score,
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
                    for s in filtered.explanation_sections
                ],
                "temporal_evolution": (
                    {
                        "confidence_trend": filtered.temporal_evolution.confidence_trend,
                        "velocity": filtered.temporal_evolution.belief_velocity_summary,
                        "stability": filtered.temporal_evolution.stability_label,
                        "undecided_duration": filtered.temporal_evolution.undecided_duration,
                    }
                    if filtered.temporal_evolution else None
                ),
            },

            # Human-readable narrative
            "human_readable": human_text,

            # Meta
            "role": role_enum.value,
        }

    return router
