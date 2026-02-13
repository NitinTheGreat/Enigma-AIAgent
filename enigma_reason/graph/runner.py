"""Graph runner — clean interface for invoking the reasoning graph.

Usage:
    from enigma_reason.graph.runner import run_reasoning

    result = run_reasoning(situation, temporal_snap, reasoning_snap)

The runner builds the graph, seeds the initial state, invokes LangGraph,
and returns the final ReasoningState.  No side effects.  No store mutation.

Phase 5.1 seeds epistemic control fields in the initial state.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from enigma_reason.config import settings
from enigma_reason.domain.reasoning import SituationReasoningSnapshot
from enigma_reason.domain.situation import Situation
from enigma_reason.domain.temporal import SituationTemporalSnapshot
from enigma_reason.graph.builder import build_reasoning_graph
from enigma_reason.graph.state import ReasoningState

logger = logging.getLogger(__name__)


def _default_llm_factory():
    """Create a Gemini Flash instance from environment config."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("ENIGMA_GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GOOGLE_API_KEY or ENIGMA_GEMINI_API_KEY "
            "in your environment variables."
        )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=api_key,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
    )


def run_reasoning(
    situation: Situation,
    temporal: SituationTemporalSnapshot,
    reasoning: SituationReasoningSnapshot,
    *,
    llm_factory=None,
    max_iterations: int | None = None,
    convergence_threshold: float | None = None,
    convergence_persistence: int | None = None,
) -> dict[str, Any]:
    """Invoke the reasoning graph for a given situation.

    Args:
        situation: The situation to reason about.
        temporal: Phase 2 temporal snapshot.
        reasoning: Phase 4 reasoning snapshot.
        llm_factory: Optional override for LLM construction (for testing).
        max_iterations: Override max loop iterations.
        convergence_threshold: Override convergence exit threshold.
        convergence_persistence: Override required dominant iterations.

    Returns:
        Final ReasoningState dict with hypotheses, convergence_score, etc.
    """
    factory = llm_factory or _default_llm_factory
    max_iter = max_iterations or settings.graph_max_iterations
    conv_threshold = convergence_threshold or settings.graph_convergence_threshold
    conv_persistence = convergence_persistence or settings.graph_convergence_persistence

    # Seed initial state — Phase 5.1 adds epistemic control fields
    initial_state: ReasoningState = {
        "situation_id": str(situation.situation_id),
        "temporal_snapshot": temporal.model_dump(),
        "reasoning_snapshot": reasoning.model_dump(),
        "context": {},
        "hypotheses": [],
        "iteration_count": 0,
        "convergence_score": 0.0,
        "max_iterations": max_iter,
        "convergence_threshold": conv_threshold,
        # Phase 5.1 epistemic controls
        "belief_stability_score": 0.0,
        "undecided_iterations": 0,
        "last_confidence_shift": 0.0,
        "convergence_persistence": conv_persistence,
    }

    # Build and invoke graph
    compiled_graph = build_reasoning_graph(factory)
    logger.info(
        "Running reasoning graph for situation %s (max_iter=%d, threshold=%.2f, persistence=%d)",
        situation.situation_id, max_iter, conv_threshold, conv_persistence,
    )

    final_state = compiled_graph.invoke(initial_state)

    logger.info(
        "Reasoning complete: situation=%s iterations=%d convergence=%.3f hypotheses=%d "
        "stability=%.3f undecided=%d",
        situation.situation_id,
        final_state.get("iteration_count", 0),
        final_state.get("convergence_score", 0.0),
        len(final_state.get("hypotheses", [])),
        final_state.get("belief_stability_score", 0.0),
        final_state.get("undecided_iterations", 0),
    )

    return final_state
