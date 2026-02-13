"""LangGraph nodes — pure functions that transform ReasoningState.

Each node:
    - Receives the full ReasoningState
    - Returns a partial dict update
    - Has no side effects beyond the LLM call in generate_hypotheses
    - Never accesses SituationStore or raw signals

LLM usage:
    generate_hypotheses uses Google Gemini Flash via langchain-google-genai.
    The LLM sees ONLY aggregated metrics — never raw signals, entity
    identifiers, or timestamps.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable
from uuid import uuid4

from enigma_reason.domain.hypothesis import Hypothesis, HypothesisStatus
from enigma_reason.graph.state import ReasoningState

logger = logging.getLogger(__name__)

# ── Type alias for LLM factory ──────────────────────────────────────────────

LLMFactory = Callable[[], Any]  # Returns a langchain BaseChatModel


# ── 1. assemble_context ─────────────────────────────────────────────────────

def assemble_context(state: ReasoningState) -> dict:
    """Build structured context from snapshots for LLM consumption.

    Exposes ONLY aggregated metrics — no raw signals, no entity identifiers,
    no timestamps, no signal IDs.  This is the information barrier.
    """
    ts = state.get("temporal_snapshot", {})
    rs = state.get("reasoning_snapshot", {})

    context = {
        "evidence_count": rs.get("evidence_count", 0),
        "event_rate_per_minute": ts.get("event_rate_per_minute", 0.0),
        "active_duration_seconds": ts.get("active_duration_seconds", 0.0),
        "burst_detected": rs.get("burst_detected", False),
        "quiet_detected": rs.get("quiet_detected", False),
        "trend": rs.get("trend", "stable"),
        "confidence_level": rs.get("confidence_level", 0.0),
        "source_diversity": rs.get("source_diversity", 0),
        "mean_anomaly_score": rs.get("mean_anomaly_score", 0.0),
        "iteration": state.get("iteration_count", 0),
    }

    logger.debug("Assembled context: %s", context)
    return {"context": context}


# ── 2. generate_hypotheses ──────────────────────────────────────────────────

_HYPOTHESIS_PROMPT = """You are a security analysis reasoning engine.

Given the following AGGREGATED situation metrics (no raw data), propose exactly 3 hypotheses about what might be happening. One hypothesis MUST be a benign/normal explanation.

Situation metrics:
- Evidence count: {evidence_count}
- Event rate: {event_rate_per_minute:.2f} events/min
- Active duration: {active_duration_seconds:.0f} seconds
- Burst detected: {burst_detected}
- Quiet detected: {quiet_detected}
- Trend: {trend}
- Confidence level: {confidence_level:.2f}
- Source diversity: {source_diversity} distinct sources
- Mean anomaly score: {mean_anomaly_score:.2f}
- Reasoning iteration: {iteration}

{existing_hypothesis_context}

Respond with ONLY a JSON array of exactly 3 objects, each with:
- "description": short neutral factual statement (10-100 chars)
- "confidence": initial confidence 0.1-0.5 (float)
- "is_benign": true if this is the benign explanation

Example format:
[
  {{"description": "Elevated activity from routine automated scanning", "confidence": 0.3, "is_benign": true}},
  {{"description": "Coordinated probing from multiple source vectors", "confidence": 0.3, "is_benign": false}},
  {{"description": "Anomalous data transfer pattern with high volume", "confidence": 0.2, "is_benign": false}}
]

RESPOND WITH ONLY THE JSON ARRAY. No markdown, no explanation."""


def _build_existing_hypothesis_context(hypotheses: list[dict]) -> str:
    """Summarise existing hypotheses for the LLM to refine."""
    if not hypotheses:
        return "No prior hypotheses exist. Generate fresh hypotheses."
    active = [h for h in hypotheses if h.get("status") == "active"]
    if not active:
        return "All prior hypotheses were pruned. Generate new hypotheses."
    lines = ["Prior active hypotheses (refine or replace):"]
    for h in active:
        lines.append(f"  - \"{h['description']}\" (confidence: {h.get('confidence', 0.0):.2f})")
    return "\n".join(lines)


def _parse_hypotheses_response(text: str) -> list[dict]:
    """Parse LLM response into hypothesis dicts, with robust fallback."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        raw = json.loads(text)
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError("Expected non-empty JSON array")

        hypotheses = []
        for item in raw[:5]:  # cap at 5
            desc = str(item.get("description", ""))[:200]
            if len(desc) < 5:
                desc = "Unspecified hypothesis from reasoning"
            conf = float(item.get("confidence", 0.3))
            conf = max(0.1, min(conf, 0.5))  # constrain initial confidence
            hypotheses.append({
                "hypothesis_id": str(uuid4()),
                "description": desc,
                "confidence": conf,
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "status": "active",
            })
        return hypotheses

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse LLM hypothesis response: %s — using fallback", exc)
        return _fallback_hypotheses()


def _fallback_hypotheses() -> list[dict]:
    """Deterministic fallback when LLM output is unparseable."""
    return [
        {
            "hypothesis_id": str(uuid4()),
            "description": "Normal operational variation in signal patterns",
            "confidence": 0.3,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "status": "active",
        },
        {
            "hypothesis_id": str(uuid4()),
            "description": "Correlated anomaly cluster from related sources",
            "confidence": 0.3,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "status": "active",
        },
        {
            "hypothesis_id": str(uuid4()),
            "description": "Transient environmental factor causing elevated scores",
            "confidence": 0.25,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "status": "active",
        },
    ]


def make_generate_hypotheses(llm_factory: LLMFactory):
    """Create the generate_hypotheses node with an injected LLM factory.

    Args:
        llm_factory: Callable that returns a langchain BaseChatModel instance.
    """

    def generate_hypotheses(state: ReasoningState) -> dict:
        """Propose hypotheses via Gemini Flash based on structured context."""
        context = state.get("context", {})
        existing = state.get("hypotheses", [])

        prompt = _HYPOTHESIS_PROMPT.format(
            **context,
            existing_hypothesis_context=_build_existing_hypothesis_context(existing),
        )

        try:
            llm = llm_factory()
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            logger.info("Gemini hypothesis response length: %d chars", len(response_text))
            hypotheses = _parse_hypotheses_response(response_text)
        except Exception as exc:
            logger.error("LLM invocation failed: %s — using fallback hypotheses", exc)
            hypotheses = _fallback_hypotheses()

        return {"hypotheses": hypotheses}

    return generate_hypotheses


# ── 3. evaluate_hypotheses ──────────────────────────────────────────────────

def evaluate_hypotheses(state: ReasoningState) -> dict:
    """Deterministically adjust hypothesis confidence using reasoning facts.

    This node is PURE deterministic logic — no LLM calls.
    It uses the reasoning snapshot to boost or penalise hypotheses.
    """
    rs = state.get("reasoning_snapshot", {})
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        return {"hypotheses": []}

    trend = rs.get("trend", "stable")
    burst = rs.get("burst_detected", False)
    quiet = rs.get("quiet_detected", False)
    confidence_level = rs.get("confidence_level", 0.0)
    mean_anomaly = rs.get("mean_anomaly_score", 0.0)

    updated = []
    for h in hypotheses:
        if h.get("status") != "active":
            updated.append(h)
            continue

        h = dict(h)  # work on a copy
        conf = h["confidence"]

        # ── Deterministic adjustments based on reasoning facts ──

        # High anomaly score boosts non-benign hypotheses
        if mean_anomaly > 0.7:
            conf += 0.1
        elif mean_anomaly < 0.3:
            conf -= 0.05

        # Burst supports escalation-related hypotheses
        if burst:
            conf += 0.1

        # Quiet reduces urgency
        if quiet:
            conf -= 0.1

        # Strong trend alignment
        if trend == "escalating":
            conf += 0.05
        elif trend == "deescalating":
            conf -= 0.05

        # Higher deterministic confidence boosts all hypotheses slightly
        conf += confidence_level * 0.1

        # Clamp
        conf = max(0.0, min(conf, 1.0))

        # Prune weak hypotheses
        if conf < 0.1:
            h["status"] = "pruned"
            logger.debug("Pruned hypothesis: %s (conf=%.3f)", h["description"], conf)

        h["confidence"] = round(conf, 4)
        updated.append(h)

    return {"hypotheses": updated}


# ── 4. update_convergence ───────────────────────────────────────────────────

def update_convergence(state: ReasoningState) -> dict:
    """Compute convergence_score from hypothesis confidence distribution.

    High convergence = one hypothesis dominates (clear signal).
    Low convergence  = ambiguity (hypotheses are close in confidence).
    """
    hypotheses = state.get("hypotheses", [])
    iteration = state.get("iteration_count", 0) + 1

    active = [h for h in hypotheses if h.get("status") == "active"]
    if not active:
        return {"convergence_score": 1.0, "iteration_count": iteration}

    confidences = [h["confidence"] for h in active]
    max_conf = max(confidences)
    mean_conf = sum(confidences) / len(confidences)

    if len(active) == 1:
        convergence = max_conf
    else:
        # Measure dominance: how far ahead is the leader?
        spread = max_conf - mean_conf
        # Normalise: if leader is 0.3 ahead of mean, convergence ≈ 0.6
        convergence = min(spread * 2.0 + max_conf * 0.3, 1.0)

    # Mark dominant hypothesis as converged if convergence is high
    if convergence >= state.get("convergence_threshold", 0.8):
        for h in hypotheses:
            if h.get("status") == "active" and h["confidence"] == max_conf:
                h["status"] = "converged"
                break

    logger.debug(
        "Convergence: %.3f (iteration %d, %d active hypotheses)",
        convergence, iteration, len(active),
    )

    return {
        "convergence_score": round(convergence, 4),
        "iteration_count": iteration,
        "hypotheses": hypotheses,
    }


# ── 5. check_convergence ───────────────────────────────────────────────────

def check_convergence(state: ReasoningState) -> str:
    """Conditional edge: decide whether to loop or exit.

    Returns:
        "end"  — if converged or max iterations reached
        "loop" — continue reasoning
    """
    convergence = state.get("convergence_score", 0.0)
    threshold = state.get("convergence_threshold", 0.8)
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if convergence >= threshold:
        logger.info("Converged at iteration %d (score=%.3f)", iteration, convergence)
        return "end"

    if iteration >= max_iter:
        logger.info("Max iterations reached (%d), stopping", max_iter)
        return "end"

    logger.debug("Not converged (score=%.3f, iter=%d), looping", convergence, iteration)
    return "loop"
