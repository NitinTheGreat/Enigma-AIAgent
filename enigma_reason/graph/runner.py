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
from enigma_reason.graph.builder import EpistemicControls, build_reasoning_graph
from enigma_reason.graph.state import ReasoningState
from enigma_reason.observability.run_log import IterationRecorder, RecordSink

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


def controls_from_settings() -> EpistemicControls:
    """Build the epistemic control set from environment-backed settings.

    Returns:
        An EpistemicControls carrying the four ablation switches and the
        belief inertia cap as configured by the ENIGMA_ environment variables.
    """
    return EpistemicControls(
        unknown_hypothesis_enabled=settings.unknown_hypothesis_enabled,
        sanity_gate_enabled=settings.sanity_gate_enabled,
        asymmetric_decay_enabled=settings.asymmetric_decay_enabled,
        persistence_required=settings.persistence_required,
        max_confidence_delta=settings.max_confidence_delta,
    )


def _invoke_with_run_log(
    compiled_graph,
    initial_state: ReasoningState,
    situation: Situation,
    run_log: RecordSink,
    run_id: str | None,
) -> dict[str, Any]:
    """Drive the graph by its value stream so each iteration can be recorded.

    LangGraph implements invoke over the same value stream this consumes, and
    the last value the stream yields is the state invoke would have returned.
    Observing here therefore leaves the reasoning path untouched: no node is
    modified, no node is aware of the recorder, and nothing the recorder does
    is visible to the graph. tests/test_level6_instrumentation.py asserts the
    two paths produce identical final states.

    A recorder failure must not fail an analysis, so the stream is drained
    inside a guard that falls back to the last state seen.
    """
    recorder = IterationRecorder(
        run_log,
        run_id=run_id,
        situation_id=str(situation.situation_id),
        clock_mode=situation.clock_mode.value,
        event_timestamp=situation.last_seen_at,
    )

    final_state: dict[str, Any] = dict(initial_state)
    for state in compiled_graph.stream(initial_state, stream_mode="values"):
        final_state = state
        try:
            recorder.observe(state)
        except Exception as exc:
            logger.error("Run log observation failed, continuing: %s", exc)

    try:
        recorder.finalise(final_state)
    except Exception as exc:
        logger.error("Run log finalisation failed, continuing: %s", exc)

    return final_state


def run_reasoning(
    situation: Situation,
    temporal: SituationTemporalSnapshot,
    reasoning: SituationReasoningSnapshot,
    *,
    llm_factory=None,
    max_iterations: int | None = None,
    convergence_threshold: float | None = None,
    convergence_persistence: int | None = None,
    controls: EpistemicControls | None = None,
    run_log: RecordSink | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the reasoning graph for a given situation.

    Args:
        situation: The situation to reason about.
        temporal: Phase 2 temporal snapshot.
        reasoning: Phase 4 reasoning snapshot.
        llm_factory: Optional override for LLM construction, used by tests.
        max_iterations: Override max loop iterations.
        convergence_threshold: Override convergence exit threshold.
        convergence_persistence: Override required dominant iterations.
        controls: Override the epistemic control set. Defaults to the values in
            settings, which Level 9 varies through environment variables.
        run_log: Optional sink receiving one record per reasoning iteration.
            When present the graph is driven by its value stream rather than by
            invoke, which observes the same execution without altering it.
        run_id: Identifier tying this analysis to the experiment that ran it.

    Returns:
        Final ReasoningState dict with hypotheses, convergence_score and the
        epistemic control set that produced it.
    """
    factory = llm_factory or _default_llm_factory
    max_iter = max_iterations or settings.graph_max_iterations
    conv_threshold = convergence_threshold or settings.graph_convergence_threshold
    conv_persistence = convergence_persistence or settings.graph_convergence_persistence
    active_controls = controls or controls_from_settings()

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
        "belief_stability_score": 0.0,
        "undecided_iterations": 0,
        "last_confidence_shift": 0.0,
        "convergence_persistence": conv_persistence,
        "max_confidence_delta": active_controls.max_confidence_delta,
    }

    compiled_graph = build_reasoning_graph(factory, controls=active_controls)
    logger.info(
        "Running reasoning graph for situation %s, max_iterations=%d, "
        "threshold=%.2f, persistence=%d, controls=%s",
        situation.situation_id,
        max_iter,
        conv_threshold,
        conv_persistence,
        active_controls.as_dict(),
    )

    if run_log is None:
        final_state = compiled_graph.invoke(initial_state)
    else:
        final_state = _invoke_with_run_log(
            compiled_graph, initial_state, situation, run_log, run_id
        )

    final_state["epistemic_controls"] = active_controls.as_dict()
    final_state["clock_mode"] = situation.clock_mode.value

    logger.info(
        "Reasoning complete for situation %s, iterations=%d convergence=%.3f "
        "hypotheses=%d stability=%.3f undecided=%d",
        situation.situation_id,
        final_state.get("iteration_count", 0),
        final_state.get("convergence_score", 0.0),
        len(final_state.get("hypotheses", [])),
        final_state.get("belief_stability_score", 0.0),
        final_state.get("undecided_iterations", 0),
    )

    return final_state
