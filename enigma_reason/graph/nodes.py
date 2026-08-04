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
    UNKNOWN_HYPOTHESIS_ID,
    make_unknown_hypothesis,
)
from enigma_reason.graph.state import ReasoningState

logger = logging.getLogger(__name__)

# ── Type alias for LLM factory ──────────────────────────────────────────────

LLMFactory = Callable[[], Any]

# ── Sanity gate tuning ──────────────────────────────────────────────────────

BENIGN_KEYWORDS: frozenset[str] = frozenset(
    {"normal", "routine", "benign", "expected", "operational", "standard"}
)
VAGUE_KEYWORDS: frozenset[str] = frozenset({"something", "maybe", "possibly", "might be"})
VAGUE_HYPOTHESIS_PENALTY: float = 0.1
MINIMUM_HYPOTHESIS_CONFIDENCE: float = 0.1

SPARSE_EVIDENCE_THRESHOLD: int = 3
MODERATE_EVIDENCE_THRESHOLD: int = 5
SPARSE_EVIDENCE_UNKNOWN_BOOST: float = 0.15
MODERATE_EVIDENCE_UNKNOWN_BOOST: float = 0.05
LOW_DIVERSITY_UNKNOWN_BOOST: float = 0.1
FLAT_SPREAD_UNKNOWN_BOOST: float = 0.1
FLAT_SPREAD_THRESHOLD: float = 0.1

# ── Evaluation tuning ───────────────────────────────────────────────────────

ASYMMETRIC_DECAY_MULTIPLIER: float = 1.5
PRUNE_CONFIDENCE_FLOOR: float = 0.1

# ── Convergence tuning ──────────────────────────────────────────────────────

UNKNOWN_DOMINANCE_MARGIN: float = 0.15
FLAT_DISTRIBUTION_PENALTY: float = 0.3
HIGH_ANOMALY_LOW_DIVERSITY_PENALTY: float = 0.5
HIGH_ANOMALY_THRESHOLD: float = 0.7
STABILITY_SHIFT_SCALE: float = 5.0

# ── Belief inertia ──────────────────────────────────────────────────────────

DEFAULT_MAX_CONFIDENCE_DELTA: float = 0.15
VELOCITY_DAMPING: float = 0.7


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
        lines = [line for line in lines if not line.strip().startswith("```")]
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


def make_generate_hypotheses(llm_factory: LLMFactory, unknown_enabled: bool = True):
    """Create the generate_hypotheses node with an injected LLM factory.

    Args:
        llm_factory: Callable returning a langchain chat model.
        unknown_enabled: Ablation switch U. When False the permanent UNKNOWN
            hypothesis is never created, so the reasoning path behaves as though
            the mechanism had not been written. Downstream nodes already treat an
            absent UNKNOWN as zero confidence, so nothing else needs to change.

    Returns:
        The generate_hypotheses node function.
    """

    def generate_hypotheses(state: ReasoningState) -> dict:
        """Propose hypotheses via the configured LLM from structured context."""
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
            logger.info("LLM hypothesis response length: %d chars", len(response_text))
            hypotheses = _parse_hypotheses_response(response_text)
        except Exception as exc:
            logger.error("LLM invocation failed: %s, using fallback hypotheses", exc)
            hypotheses = _fallback_hypotheses()

        if not unknown_enabled:
            return {"hypotheses": [
                h for h in hypotheses if h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID
            ]}

        unknown_exists = any(
            h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID for h in hypotheses
        )
        if not unknown_exists:
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
        h = dict(h)
        key = h.get("description", "").lower().strip()[:50]
        if key in seen_descriptions and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID:
            h["status"] = "pruned"
            logger.debug("Sanity gate pruned duplicate hypothesis: %s", h["description"])
        else:
            seen_descriptions.add(key)
        deduped.append(h)

    # ── Ensure at least one benign-sounding hypothesis ───────────────────
    active_non_unknown = [
        h for h in deduped
        if h.get("status") == "active"
        and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID
    ]
    has_benign = any(
        any(keyword in h.get("description", "").lower() for keyword in BENIGN_KEYWORDS)
        for h in active_non_unknown
    )
    if not has_benign and active_non_unknown:
        # Inject a benign hypothesis
        deduped.append(_new_hypothesis_dict(
            "Normal operational activity within expected parameters", 0.25,
        ))

    # ── Penalise vague hypotheses ────────────────────────────────────────
    for index, h in enumerate(deduped):
        if h.get("status") != "active" or h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID:
            continue
        desc_lower = h.get("description", "").lower()
        if any(keyword in desc_lower for keyword in VAGUE_KEYWORDS):
            before = h.get("confidence", 0.3)
            after = max(MINIMUM_HYPOTHESIS_CONFIDENCE, before - VAGUE_HYPOTHESIS_PENALTY)
            deduped[index] = {**h, "confidence": round(after, 4)}
            logger.debug(
                "Sanity gate penalised vague hypothesis: %s confidence %.4f to %.4f",
                h.get("description"),
                before,
                after,
            )

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

        if evidence_count < SPARSE_EVIDENCE_THRESHOLD:
            unknown_h["confidence"] = min(
                1.0, unknown_h["confidence"] + SPARSE_EVIDENCE_UNKNOWN_BOOST
            )
        elif evidence_count < MODERATE_EVIDENCE_THRESHOLD:
            unknown_h["confidence"] = min(
                1.0, unknown_h["confidence"] + MODERATE_EVIDENCE_UNKNOWN_BOOST
            )

        if source_diversity <= 1:
            unknown_h["confidence"] = min(
                1.0, unknown_h["confidence"] + LOW_DIVERSITY_UNKNOWN_BOOST
            )

        active_confs = [
            h["confidence"] for h in deduped
            if h.get("status") == "active" and h.get("hypothesis_id") != UNKNOWN_HYPOTHESIS_ID
        ]
        if len(active_confs) >= 2:
            spread = max(active_confs) - min(active_confs)
            if spread < FLAT_SPREAD_THRESHOLD:
                unknown_h["confidence"] = min(
                    1.0, unknown_h["confidence"] + FLAT_SPREAD_UNKNOWN_BOOST
                )

        deduped[idx] = unknown_h

    return {"hypotheses": deduped}


# ── 4. evaluate_hypotheses ──────────────────────────────────────────────────

def make_evaluate_hypotheses(asymmetric_decay_enabled: bool = True):
    """Create the evaluate_hypotheses node.

    Args:
        asymmetric_decay_enabled: Ablation switch A. When False a negative
            adjustment is applied at face value rather than multiplied, which is
            exactly the behaviour of the node before the mechanism existed.

    Returns:
        The evaluate_hypotheses node function.
    """

    def evaluate_hypotheses(state: ReasoningState) -> dict:
        """Deterministically adjust hypothesis confidence using reasoning facts.

        Pure deterministic logic, no LLM calls. Records the pre-adjustment
        confidence on each hypothesis so the inertia node can compute a real
        per-iteration delta.

        Args:
            state: The current reasoning state.

        Returns:
            A partial state update carrying the adjusted hypotheses and the
            largest absolute confidence shift observed in this pass.
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
        for original in hypotheses:
            if original.get("status") != "active":
                updated.append(dict(original))
                continue

            h = dict(original)
            old_conf = h["confidence"]
            conf = old_conf

            is_unknown = h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID

            if not is_unknown:
                if mean_anomaly > HIGH_ANOMALY_THRESHOLD:
                    conf += 0.1
                elif mean_anomaly < 0.3:
                    conf -= 0.05

                if burst:
                    conf += 0.1

                if quiet:
                    conf -= 0.1

                if trend == "escalating":
                    conf += 0.05
                elif trend == "deescalating":
                    conf -= 0.05

                conf += confidence_level * 0.1

                delta = conf - old_conf
                if asymmetric_decay_enabled and delta < 0:
                    conf = old_conf + delta * ASYMMETRIC_DECAY_MULTIPLIER

            conf = max(0.0, min(conf, 1.0))

            if conf < PRUNE_CONFIDENCE_FLOOR and not is_unknown:
                h["status"] = "pruned"
                logger.debug("Pruned hypothesis: %s at confidence %.3f", h["description"], conf)

            h["confidence_before_inertia"] = round(conf, 4)
            h["confidence_previous"] = round(old_conf, 4)
            h["confidence"] = round(conf, 4)
            max_shift = max(max_shift, abs(conf - old_conf))
            updated.append(h)

        return {"hypotheses": updated, "last_confidence_shift": round(max_shift, 4)}

    return evaluate_hypotheses


evaluate_hypotheses = make_evaluate_hypotheses()


# ── 5. apply_belief_inertia ─────────────────────────────────────────────────

def make_apply_belief_inertia(max_confidence_delta: float = DEFAULT_MAX_CONFIDENCE_DELTA):
    """Create the apply_belief_inertia node.

    Args:
        max_confidence_delta: Largest absolute confidence change permitted in a
            single iteration. Set to a very large number to disable the
            mechanism, which makes the node a no-op on confidence and therefore
            behaviourally identical to the node not being present. This argument
            is authoritative; the identically named state field is carried for
            run manifests and logging only, and never overrides it.

    Returns:
        The apply_belief_inertia node function.
    """

    def apply_belief_inertia(state: ReasoningState) -> dict:
        """Rate-limit per-iteration confidence change.

        The evaluation node records the pre-adjustment confidence on each
        hypothesis. This node compares the two, clamps the change to
        max_confidence_delta, and records both the pre-clamp and post-clamp
        values so the clamp is auditable from the run log.

        Args:
            state: The current reasoning state.

        Returns:
            A partial state update carrying the clamped hypotheses.
        """
        hypotheses = state.get("hypotheses", [])
        if not hypotheses:
            return {"hypotheses": []}

        cap = max_confidence_delta

        updated = []
        for original in hypotheses:
            if original.get("status") != "active":
                updated.append(dict(original))
                continue

            h = dict(original)
            previous = h.get("confidence_previous", h.get("confidence", 0.3))
            proposed = h.get("confidence_before_inertia", h.get("confidence", 0.3))

            raw_delta = proposed - previous
            if abs(raw_delta) > cap:
                clamped_delta = cap if raw_delta > 0 else -cap
                logger.debug(
                    "Belief inertia clamped %s from delta %.4f to %.4f",
                    h.get("hypothesis_id"),
                    raw_delta,
                    clamped_delta,
                )
            else:
                clamped_delta = raw_delta

            applied = max(0.0, min(previous + clamped_delta, 1.0))
            old_velocity = h.get("belief_velocity", 0.0)
            new_velocity = (
                old_velocity * VELOCITY_DAMPING + (1.0 - VELOCITY_DAMPING) * clamped_delta
            )

            h["confidence_before_inertia"] = round(proposed, 4)
            h["confidence"] = round(applied, 4)
            h["belief_velocity"] = round(new_velocity, 4)
            h["belief_acceleration"] = round(new_velocity - old_velocity, 4)
            h["inertia_clamped"] = abs(raw_delta) > cap
            updated.append(h)

        return {"hypotheses": updated}

    return apply_belief_inertia


apply_belief_inertia = make_apply_belief_inertia()


# ── 6. update_convergence ───────────────────────────────────────────────────

def make_update_convergence(persistence_required: bool = True):
    """Create the update_convergence node.

    Args:
        persistence_required: Ablation switch P. When False a leader may
            converge on the iteration it first takes the lead, which is the
            behaviour before the sustained dominance requirement existed.

    Returns:
        The update_convergence node function.
    """

    def update_convergence(state: ReasoningState) -> dict:
        """Compute convergence_score from the hypothesis confidence distribution.

        This node is pure. It never mutates the hypotheses it is given; every
        returned hypothesis is a new dict. Mutating shared state here made the
        result depend on evaluation order, which is recorded as defect D4 in
        paper/EVIDENCE.md.

        Args:
            state: The current reasoning state.

        Returns:
            A partial state update carrying a new hypothesis list, the
            convergence score, the iteration count, the belief stability score
            and the undecided iteration counter.
        """
        source = state.get("hypotheses", [])
        hypotheses = [dict(h) for h in source]
        iteration = state.get("iteration_count", 0) + 1
        required_persistence = state.get("convergence_persistence", 2)
        threshold = state.get("convergence_threshold", 0.8)

        active = [h for h in hypotheses if h.get("status") == "active"]
        if not active:
            return {
                "convergence_score": 0.0,
                "iteration_count": iteration,
                "hypotheses": hypotheses,
                "belief_stability_score": 0.0,
                "undecided_iterations": state.get("undecided_iterations", 0) + 1,
            }

        confidences = [h["confidence"] for h in active]
        max_conf = max(confidences)
        mean_conf = sum(confidences) / len(confidences)

        dominant = max(active, key=lambda h: h["confidence"])
        dominant_id = dominant.get("hypothesis_id")

        unknown = next(
            (h for h in active if h.get("hypothesis_id") == UNKNOWN_HYPOTHESIS_ID), None,
        )
        unknown_conf = unknown["confidence"] if unknown else 0.0

        if dominant_id == UNKNOWN_HYPOTHESIS_ID:
            convergence = 0.0
            undecided = state.get("undecided_iterations", 0) + 1
            for h in active:
                h["dominant_iterations"] = 0
        else:
            margin_over_unknown = max_conf - unknown_conf
            if margin_over_unknown < UNKNOWN_DOMINANCE_MARGIN:
                convergence = margin_over_unknown * 2.0
                undecided = state.get("undecided_iterations", 0) + 1
            else:
                undecided = 0

            if len(active) == 1:
                convergence = max_conf
            else:
                spread = max_conf - mean_conf
                convergence = min(spread * 2.0 + margin_over_unknown * 0.5, 1.0)

            if len(active) >= 2:
                conf_range = max(confidences) - min(confidences)
                if conf_range < FLAT_SPREAD_THRESHOLD:
                    convergence *= FLAT_DISTRIBUTION_PENALTY

            rs = state.get("reasoning_snapshot", {})
            if (
                rs.get("mean_anomaly_score", 0.0) > HIGH_ANOMALY_THRESHOLD
                and rs.get("source_diversity", 0) <= 1
            ):
                convergence *= HIGH_ANOMALY_LOW_DIVERSITY_PENALTY

            for h in active:
                if h.get("hypothesis_id") == dominant_id:
                    h["dominant_iterations"] = h.get("dominant_iterations", 0) + 1
                else:
                    h["dominant_iterations"] = 0

        dominant_iters = dominant.get("dominant_iterations", 0)
        persistence_satisfied = (not persistence_required) or (
            dominant_iters >= required_persistence
        )

        if (
            persistence_required
            and not persistence_satisfied
            and dominant_id != UNKNOWN_HYPOTHESIS_ID
        ):
            convergence = min(convergence, threshold - 0.01)

        last_shift = state.get("last_confidence_shift", 0.0)
        stability = max(0.0, 1.0 - last_shift * STABILITY_SHIFT_SCALE)

        if (
            convergence >= threshold
            and dominant_id != UNKNOWN_HYPOTHESIS_ID
            and persistence_satisfied
        ):
            for h in hypotheses:
                if h.get("status") == "active" and h.get("hypothesis_id") == dominant_id:
                    h["status"] = "converged"
                    break

        logger.debug(
            "Convergence %.3f at iteration %d, %d active, dominant %s, "
            "dominant_iterations %d, stability %.2f",
            convergence, iteration, len(active), dominant_id, dominant_iters, stability,
        )

        return {
            "convergence_score": round(max(0.0, min(convergence, 1.0)), 4),
            "iteration_count": iteration,
            "hypotheses": hypotheses,
            "belief_stability_score": round(stability, 4),
            "undecided_iterations": undecided,
        }

    return update_convergence


update_convergence = make_update_convergence()


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
