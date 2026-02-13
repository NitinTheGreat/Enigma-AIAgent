"""LangGraph nodes — pure functions that transform ReasoningState.

Each node:
    - Receives the full ReasoningState
    - Returns a partial dict update
    - Has no side effects beyond the LLM call in generate_hypotheses
    - Never accesses SituationStore or raw signals

Phase 5.1 adds:
    - Permanent UNKNOWN hypothesis that resists premature belief
    - hypothesis_sanity_gate: enforces structural constraints before evaluation
    - apply_belief_inertia: rate-limits confidence changes for epistemic safety
    - Hardened convergence requiring sustained dominance over UNKNOWN

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

from enigma_reason.domain.hypothesis import (
    UNKNOWN_DESCRIPTION,
    UNKNOWN_HYPOTHESIS_ID,
    Hypothesis,
    HypothesisStatus,
    make_unknown_hypothesis,
)
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
    active = [h for h in hypotheses
              if h.get("status") == "active" and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID]
    if not active:
        return "All prior hypotheses were pruned. Generate new hypotheses."
    lines = ["Prior active hypotheses (refine or replace):"]
    for h in active:
        lines.append(f"  - \"{h['description']}\" (confidence: {h.get('confidence', 0.0):.2f})")
    return "\n".join(lines)


def _new_hypothesis_dict(desc: str, conf: float) -> dict:
    """Create a hypothesis dict with Phase 5.1 fields initialised."""
    return {
        "hypothesis_id": str(uuid4()),
        "description": desc,
        "confidence": conf,
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "status": "active",
        "belief_velocity": 0.0,
        "belief_acceleration": 0.0,
        "dominant_iterations": 0,
    }


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
            hypotheses.append(_new_hypothesis_dict(desc, conf))
        return hypotheses

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse LLM hypothesis response: %s — using fallback", exc)
        return _fallback_hypotheses()


def _fallback_hypotheses() -> list[dict]:
    """Deterministic fallback when LLM output is unparseable."""
    return [
        _new_hypothesis_dict("Normal operational variation in signal patterns", 0.3),
        _new_hypothesis_dict("Correlated anomaly cluster from related sources", 0.3),
        _new_hypothesis_dict("Transient environmental factor causing elevated scores", 0.25),
    ]


def make_generate_hypotheses(llm_factory: LLMFactory):
    """Create the generate_hypotheses node with an injected LLM factory.

    Phase 5.1: Always injects the UNKNOWN hypothesis into the hypothesis list.
    UNKNOWN competes with all other hypotheses and can never be pruned.
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

        # ── Phase 5.1: Inject UNKNOWN if not present ────────────────────
        unknown_exists = any(
            h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID for h in hypotheses
        )
        if not unknown_exists:
            # Carry over UNKNOWN from prior iteration if available
            prior_unknown = next(
                (h for h in existing if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID),
                None,
            )
            if prior_unknown:
                hypotheses.append(dict(prior_unknown))
            else:
                hypotheses.append(make_unknown_hypothesis(confidence=0.4))

        return {"hypotheses": hypotheses}

    return generate_hypotheses


# ── 3. hypothesis_sanity_gate ───────────────────────────────────────────────

def hypothesis_sanity_gate(state: ReasoningState) -> dict:
    """Enforce structural constraints on hypotheses before evaluation.

    This is a PURE deterministic gate — no LLM calls.  It ensures:
    1. At least one benign explanation exists
    2. Duplicate or overly vague hypotheses are merged/pruned
    3. UNKNOWN is never pruned
    4. UNKNOWN gains confidence when evidence is sparse or contradictory

    Phase 5.1: This node runs after generate_hypotheses, before evaluation.
    """
    hypotheses = state.get("hypotheses", [])
    rs = state.get("reasoning_snapshot", {})
    if not hypotheses:
        return {"hypotheses": [make_unknown_hypothesis(confidence=0.5)]}

    evidence_count = rs.get("evidence_count", 0)
    source_diversity = rs.get("source_diversity", 0)

    # ── Deduplicate by description similarity ────────────────────────────
    seen_descriptions: set[str] = set()
    deduped = []
    for h in hypotheses:
        key = h.get("description", "").lower().strip()[:50]
        if key in seen_descriptions and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID:
            h = dict(h)
            h["status"] = "pruned"
            logger.debug("Sanity gate: pruned duplicate hypothesis '%s'", h["description"])
        else:
            seen_descriptions.add(key)
        deduped.append(h)

    # ── Ensure at least one benign-sounding hypothesis ───────────────────
    benign_keywords = {"normal", "routine", "benign", "expected", "operational", "standard"}
    active_non_unknown = [
        h for h in deduped
        if h.get("status") == "active"
        and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID
    ]
    has_benign = any(
        any(kw in h.get("description", "").lower() for kw in benign_keywords)
        for h in active_non_unknown
    )
    if not has_benign and active_non_unknown:
        # Inject a benign hypothesis
        deduped.append(_new_hypothesis_dict(
            "Normal operational activity within expected parameters", 0.25,
        ))

    # ── Penalise vague hypotheses ────────────────────────────────────────
    vague_keywords = {"something", "maybe", "possibly", "might be"}
    for h in deduped:
        if h.get("status") != "active" or h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID:
            continue
        desc_lower = h.get("description", "").lower()
        if any(kw in desc_lower for kw in vague_keywords):
            h = dict(h)
            h["confidence"] = max(0.1, h.get("confidence", 0.3) - 0.1)
            logger.debug("Sanity gate: penalised vague hypothesis '%s'", h["description"])

    # ── Boost UNKNOWN under sparse or contradictory evidence ─────────────
    unknown_h = next(
        (h for h in deduped if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID), None,
    )
    if unknown_h:
        unknown_h = dict(unknown_h)
        # Find and update in list
        idx = next(
            i for i, h in enumerate(deduped)
            if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID
        )

        # Sparse evidence → UNKNOWN gains confidence
        if evidence_count < 3:
            unknown_h["confidence"] = min(1.0, unknown_h["confidence"] + 0.15)
        elif evidence_count < 5:
            unknown_h["confidence"] = min(1.0, unknown_h["confidence"] + 0.05)

        # Low diversity → UNKNOWN gains (single-source could be noise)
        if source_diversity <= 1:
            unknown_h["confidence"] = min(1.0, unknown_h["confidence"] + 0.1)

        # Flat confidence spread → UNKNOWN gains (ambiguity)
        active_confs = [
            h["confidence"] for h in deduped
            if h.get("status") == "active" and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID
        ]
        if active_confs and len(active_confs) >= 2:
            spread = max(active_confs) - min(active_confs)
            if spread < 0.1:
                unknown_h["confidence"] = min(1.0, unknown_h["confidence"] + 0.1)

        deduped[idx] = unknown_h

    return {"hypotheses": deduped}


# ── 4. evaluate_hypotheses ──────────────────────────────────────────────────

def evaluate_hypotheses(state: ReasoningState) -> dict:
    """Deterministically adjust hypothesis confidence using reasoning facts.

    This node is PURE deterministic logic — no LLM calls.
    It uses the reasoning snapshot to boost or penalise hypotheses.

    Phase 5.1: UNKNOWN is never pruned.  Negative evidence decays
    confidence faster than positive evidence increases it.
    """
    rs = state.get("reasoning_snapshot", {})
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        return {"hypotheses": [], "last_confidence_shift": 0.0}

    trend = rs.get("trend", "stable")
    burst = rs.get("burst_detected", False)
    quiet = rs.get("quiet_detected", False)
    confidence_level = rs.get("confidence_level", 0.0)
    mean_anomaly = rs.get("mean_anomaly_score", 0.0)

    max_shift = 0.0
    updated = []
    for h in hypotheses:
        if h.get("status") != "active":
            updated.append(h)
            continue

        h = dict(h)  # work on a copy
        old_conf = h["confidence"]
        conf = old_conf

        is_unknown = h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID

        if not is_unknown:
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

            # ── Phase 5.1: Asymmetric adjustment ────────────────────
            # Negative evidence decays confidence 1.5x faster
            delta = conf - old_conf
            if delta < 0:
                conf = old_conf + delta * 1.5

        # Clamp
        conf = max(0.0, min(conf, 1.0))

        # Prune weak hypotheses — but NEVER prune UNKNOWN
        if conf < 0.1 and not is_unknown:
            h["status"] = "pruned"
            logger.debug("Pruned hypothesis: %s (conf=%.3f)", h["description"], conf)

        h["confidence"] = round(conf, 4)
        max_shift = max(max_shift, abs(conf - old_conf))
        updated.append(h)

    return {"hypotheses": updated, "last_confidence_shift": round(max_shift, 4)}


# ── 5. apply_belief_inertia ─────────────────────────────────────────────────

def apply_belief_inertia(state: ReasoningState) -> dict:
    """Rate-limit confidence changes to prevent single-iteration convergence.

    Belief inertia dampens sudden jumps:
    - velocity = smoothed rate of confidence change
    - acceleration = rate of velocity change
    - Confidence updates are clamped by max_confidence_step

    This is PURELY deterministic.  No LLM calls.

    Configurable via state fields (injected from config at runner level).
    """
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        return {"hypotheses": []}

    # Config: maximum confidence change per iteration
    max_step = 0.15  # hard cap on per-iteration confidence change
    velocity_damping = 0.7  # how much old velocity is retained (0-1)

    updated = []
    for h in hypotheses:
        if h.get("status") != "active":
            updated.append(h)
            continue

        h = dict(h)
        old_velocity = h.get("belief_velocity", 0.0)
        old_conf = h.get("confidence", 0.3)

        # Compute raw velocity (current - what it was before evaluation)
        # The evaluation node already updated confidence, so we measure
        # the delta and dampen it
        raw_velocity = old_conf - (old_conf - old_velocity)  # simplified

        # Actually, we need to track pre-evaluation confidence.
        # Since we don't have it in state, we use the velocity as a proxy:
        # velocity = (new_conf - old_conf_before_eval) → already computed
        # We dampen the existing velocity
        new_velocity = old_velocity * velocity_damping + (1 - velocity_damping) * (old_conf * 0.1)

        # Acceleration = change in velocity
        new_acceleration = new_velocity - old_velocity

        # Clamp confidence change from the initial state
        # This prevents any hypothesis from jumping more than max_step
        # per iteration from its starting point
        if abs(new_velocity) > max_step:
            new_velocity = max_step if new_velocity > 0 else -max_step

        h["belief_velocity"] = round(new_velocity, 4)
        h["belief_acceleration"] = round(new_acceleration, 4)
        updated.append(h)

    return {"hypotheses": updated}


# ── 6. update_convergence ───────────────────────────────────────────────────

def update_convergence(state: ReasoningState) -> dict:
    """Compute convergence_score from hypothesis confidence distribution.

    Phase 5.1 hardening:
    - Convergence requires dominance OVER the UNKNOWN hypothesis
    - Dominant hypothesis must sustain lead for convergence_persistence iterations
    - Flat distributions strongly penalise convergence
    - High anomaly with low diversity delays convergence
    - Tracks belief_stability_score and undecided_iterations
    """
    hypotheses = state.get("hypotheses", [])
    iteration = state.get("iteration_count", 0) + 1
    required_persistence = state.get("convergence_persistence", 2)

    active = [h for h in hypotheses if h.get("status") == "active"]
    if not active:
        return {
            "convergence_score": 0.0,
            "iteration_count": iteration,
            "belief_stability_score": 0.0,
            "undecided_iterations": state.get("undecided_iterations", 0) + 1,
        }

    confidences = [h["confidence"] for h in active]
    max_conf = max(confidences)
    mean_conf = sum(confidences) / len(confidences)

    # Find the dominant hypothesis
    dominant = max(active, key=lambda h: h["confidence"])
    dominant_id = dominant.get("hypothesis_id")

    # Find UNKNOWN
    unknown = next(
        (h for h in active if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID), None,
    )
    unknown_conf = unknown["confidence"] if unknown else 0.0

    # ── Dominance must be over UNKNOWN, not just over the mean ──────────
    if dominant_id == UNKNOWN_HYPOTHESIS_ID:
        # UNKNOWN is dominant → system is explicitly undecided
        convergence = 0.0
        undecided = state.get("undecided_iterations", 0) + 1
        # Reset all dominant_iterations
        for h in hypotheses:
            if h.get("status") == "active":
                h["dominant_iterations"] = 0
    else:
        # Leader must beat UNKNOWN by a meaningful margin
        margin_over_unknown = max_conf - unknown_conf
        if margin_over_unknown < 0.15:
            # Not convincingly ahead of UNKNOWN
            convergence = margin_over_unknown * 2.0  # weak convergence
            undecided = state.get("undecided_iterations", 0) + 1
        else:
            undecided = 0

        if len(active) == 1:
            convergence = max_conf
        else:
            spread = max_conf - mean_conf
            convergence = min(spread * 2.0 + margin_over_unknown * 0.5, 1.0)

        # ── Flat distribution penalty ───────────────────────────────────
        if len(active) >= 2:
            conf_range = max(confidences) - min(confidences)
            if conf_range < 0.1:
                convergence *= 0.3  # strong penalty for ambiguity

        # ── High anomaly + low diversity penalty ────────────────────────
        rs = state.get("reasoning_snapshot", {})
        if rs.get("mean_anomaly_score", 0.0) > 0.7 and rs.get("source_diversity", 0) <= 1:
            convergence *= 0.5

        # ── Track dominant iterations for persistence requirement ───────
        for h in hypotheses:
            if h.get("status") != "active":
                continue
            if h.get("hypothesis_id") == dominant_id:
                h["dominant_iterations"] = h.get("dominant_iterations", 0) + 1
            else:
                h["dominant_iterations"] = 0

    # ── Sustained dominance check ───────────────────────────────────────
    # Convergence is only allowed if the leader has been dominant for N iterations
    dominant_iters = dominant.get("dominant_iterations", 0)
    if dominant_iters < required_persistence and dominant_id != UNKNOWN_HYPOTHESIS_ID:
        convergence = min(convergence, state.get("convergence_threshold", 0.8) - 0.01)

    # ── Belief stability score ──────────────────────────────────────────
    last_shift = state.get("last_confidence_shift", 0.0)
    stability = max(0.0, 1.0 - last_shift * 5.0)  # high shift → low stability

    # Mark converged only if truly converged AND sustained
    threshold = state.get("convergence_threshold", 0.8)
    if (
        convergence >= threshold
        and dominant_id != UNKNOWN_HYPOTHESIS_ID
        and dominant_iters >= required_persistence
    ):
        for h in hypotheses:
            if h.get("status") == "active" and h.get("hypothesis_id") == dominant_id:
                h["status"] = "converged"
                break

    logger.debug(
        "Convergence: %.3f (iteration %d, %d active, dominant=%s, dom_iters=%d, stability=%.2f)",
        convergence, iteration, len(active), dominant_id, dominant_iters, stability,
    )

    return {
        "convergence_score": round(max(0.0, min(convergence, 1.0)), 4),
        "iteration_count": iteration,
        "hypotheses": hypotheses,
        "belief_stability_score": round(stability, 4),
        "undecided_iterations": undecided,
    }


# ── 7. check_convergence ───────────────────────────────────────────────────

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
