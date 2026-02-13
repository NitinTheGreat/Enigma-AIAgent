"""LangGraph-based agentic reasoning graph.

This file wires together the reasoning agents into a stateful graph.
Each node represents an agent step; edges define the control flow.
No business logic is implemented here — this is a structural placeholder.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from enigma_reason.models.decision import ActionType


# ── Graph State ──────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """The state object threaded through every node in the graph."""

    raw_signal: dict[str, Any]
    threat_level: str
    reasoning_trace: list[str]
    decision_action: str | None
    decision_summary: str | None
    should_escalate: bool


# ── Node Functions (placeholders) ────────────────────────────────────────────

async def ingest_signal(state: GraphState) -> GraphState:
    """Parse and validate the incoming signal."""
    state["reasoning_trace"].append("Signal ingested")
    return state


async def assess_threat(state: GraphState) -> GraphState:
    """Evaluate threat level from signal + historical context."""
    state["reasoning_trace"].append("Threat assessed")
    state["threat_level"] = "elevated"  # placeholder
    return state


async def decide_action(state: GraphState) -> GraphState:
    """Choose a response action based on the assessed threat."""
    state["reasoning_trace"].append("Action decided")
    state["decision_action"] = ActionType.MONITOR.value  # placeholder
    state["decision_summary"] = "Monitoring — no immediate risk."
    return state


async def escalate(state: GraphState) -> GraphState:
    """Handle escalation when threat exceeds threshold."""
    state["reasoning_trace"].append("Escalated to operator")
    state["decision_action"] = ActionType.ESCALATE.value
    state["decision_summary"] = "Threat escalated to human operator."
    return state


# ── Conditional Edge ─────────────────────────────────────────────────────────

def needs_escalation(state: GraphState) -> str:
    """Route to escalation or end based on threat level."""
    if state.get("should_escalate"):
        return "escalate"
    return "end"


# ── Build Graph ──────────────────────────────────────────────────────────────

def build_reasoning_graph() -> StateGraph:
    """Construct and compile the reasoning graph."""

    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("ingest_signal", ingest_signal)
    graph.add_node("assess_threat", assess_threat)
    graph.add_node("decide_action", decide_action)
    graph.add_node("escalate", escalate)

    # Define edges
    graph.set_entry_point("ingest_signal")
    graph.add_edge("ingest_signal", "assess_threat")
    graph.add_edge("assess_threat", "decide_action")

    # Conditional branch after decision
    graph.add_conditional_edges(
        "decide_action",
        needs_escalation,
        {"escalate": "escalate", "end": END},
    )
    graph.add_edge("escalate", END)

    return graph.compile()
