"""ReasoningState — the sole state object that LangGraph nodes read and write.

This TypedDict defines the contract for the reasoning graph.  Every node
receives the full state and returns a partial update.  No node may access
anything outside this state (no store, no globals, no I/O beyond the LLM).
"""

from __future__ import annotations

from typing import Any, TypedDict


class ReasoningState(TypedDict, total=False):
    """LangGraph state for the reasoning orchestration loop.

    Fields:
        situation_id: UUID of the situation being reasoned about.
        temporal_snapshot: Serialised SituationTemporalSnapshot (dict).
        reasoning_snapshot: Serialised SituationReasoningSnapshot (dict).
        context: Structured context assembled for LLM consumption.
        hypotheses: List of serialised Hypothesis dicts.
        iteration_count: How many reasoning loops have executed.
        convergence_score: 0.0–1.0, how confident the graph is in its result.
        max_iterations: Safety cap on loop count.
        convergence_threshold: Score at which the graph stops iterating.
    """

    situation_id: str
    temporal_snapshot: dict[str, Any]
    reasoning_snapshot: dict[str, Any]
    context: dict[str, Any]
    hypotheses: list[dict[str, Any]]
    iteration_count: int
    convergence_score: float
    max_iterations: int
    convergence_threshold: float
