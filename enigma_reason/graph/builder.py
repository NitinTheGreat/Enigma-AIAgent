"""Graph builder — constructs the LangGraph reasoning topology.

Topology:

    START → assemble_context → generate_hypotheses → evaluate_hypotheses
          → update_convergence → check_convergence
                                  ├── "end"  → END
                                  └── "loop" → assemble_context

The graph is compiled once and can be invoked many times.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from enigma_reason.graph.nodes import (
    LLMFactory,
    assemble_context,
    check_convergence,
    evaluate_hypotheses,
    make_generate_hypotheses,
    update_convergence,
)
from enigma_reason.graph.state import ReasoningState


def build_reasoning_graph(llm_factory: LLMFactory) -> StateGraph:
    """Construct and compile the reasoning graph.

    Args:
        llm_factory: Callable returning a langchain BaseChatModel
                     (e.g. ChatGoogleGenerativeAI for Gemini Flash).

    Returns:
        A compiled LangGraph application.
    """
    graph = StateGraph(ReasoningState)

    # ── Register nodes ───────────────────────────────────────────────────
    graph.add_node("assemble_context", assemble_context)
    graph.add_node("generate_hypotheses", make_generate_hypotheses(llm_factory))
    graph.add_node("evaluate_hypotheses", evaluate_hypotheses)
    graph.add_node("update_convergence", update_convergence)

    # ── Edges ────────────────────────────────────────────────────────────
    graph.add_edge(START, "assemble_context")
    graph.add_edge("assemble_context", "generate_hypotheses")
    graph.add_edge("generate_hypotheses", "evaluate_hypotheses")
    graph.add_edge("evaluate_hypotheses", "update_convergence")

    # ── Conditional exit ─────────────────────────────────────────────────
    graph.add_conditional_edges(
        "update_convergence",
        check_convergence,
        {
            "end": END,
            "loop": "assemble_context",
        },
    )

    return graph.compile()
