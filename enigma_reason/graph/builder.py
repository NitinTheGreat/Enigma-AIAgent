"""
Module: enigma_reason/graph/builder.py

Constructs the LangGraph reasoning topology.

Topology with every mechanism enabled:

    START -> assemble_context -> generate_hypotheses -> hypothesis_sanity_gate
          -> evaluate_hypotheses -> apply_belief_inertia -> update_convergence
          -> check_convergence
               "end"  -> END
               "loop" -> assemble_context

Two of the four ablation switches remove a node from the topology outright
rather than neutering it, so that a disabled configuration is byte for byte the
graph that would exist had the mechanism never been written. The remaining two
change behaviour inside a node that must run regardless.

The graph is compiled once and can be invoked many times.
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from enigma_reason.graph.nodes import (
    LLMFactory,
    assemble_context,
    check_convergence,
    hypothesis_sanity_gate,
    make_apply_belief_inertia,
    make_evaluate_hypotheses,
    make_generate_hypotheses,
    make_update_convergence,
)
from enigma_reason.graph.state import ReasoningState

INERTIA_DISABLED_SENTINEL: float = float("inf")


@dataclass(frozen=True)
class EpistemicControls:
    """The four ablatable epistemic mechanisms plus the inertia cap.

    Level 9 runs a 2^4 factorial over the four booleans. Each False value must
    remove the mechanism from the reasoning path entirely.
    """

    unknown_hypothesis_enabled: bool = True
    sanity_gate_enabled: bool = True
    asymmetric_decay_enabled: bool = True
    persistence_required: bool = True
    max_confidence_delta: float = 0.15

    @property
    def inertia_enabled(self) -> bool:
        """Return False when the cap is too large ever to bind."""
        return self.max_confidence_delta < INERTIA_DISABLED_SENTINEL

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view for run manifests and logging."""
        return {
            "unknown_hypothesis_enabled": self.unknown_hypothesis_enabled,
            "sanity_gate_enabled": self.sanity_gate_enabled,
            "asymmetric_decay_enabled": self.asymmetric_decay_enabled,
            "persistence_required": self.persistence_required,
            "max_confidence_delta": self.max_confidence_delta,
            "inertia_enabled": self.inertia_enabled,
        }


def build_reasoning_graph(
    llm_factory: LLMFactory,
    controls: EpistemicControls | None = None,
) -> StateGraph:
    """Construct and compile the reasoning graph.

    Args:
        llm_factory: Callable returning a langchain BaseChatModel.
        controls: Which epistemic mechanisms are active. Defaults to all on.

    Returns:
        A compiled LangGraph application.
    """
    active = controls or EpistemicControls()

    graph = StateGraph(ReasoningState)

    graph.add_node("assemble_context", assemble_context)
    graph.add_node(
        "generate_hypotheses",
        make_generate_hypotheses(
            llm_factory, unknown_enabled=active.unknown_hypothesis_enabled
        ),
    )
    if active.sanity_gate_enabled:
        graph.add_node("hypothesis_sanity_gate", hypothesis_sanity_gate)
    graph.add_node(
        "evaluate_hypotheses",
        make_evaluate_hypotheses(asymmetric_decay_enabled=active.asymmetric_decay_enabled),
    )
    if active.inertia_enabled:
        graph.add_node(
            "apply_belief_inertia",
            make_apply_belief_inertia(max_confidence_delta=active.max_confidence_delta),
        )
    graph.add_node(
        "update_convergence",
        make_update_convergence(persistence_required=active.persistence_required),
    )

    graph.add_edge(START, "assemble_context")
    graph.add_edge("assemble_context", "generate_hypotheses")

    if active.sanity_gate_enabled:
        graph.add_edge("generate_hypotheses", "hypothesis_sanity_gate")
        graph.add_edge("hypothesis_sanity_gate", "evaluate_hypotheses")
    else:
        graph.add_edge("generate_hypotheses", "evaluate_hypotheses")

    if active.inertia_enabled:
        graph.add_edge("evaluate_hypotheses", "apply_belief_inertia")
        graph.add_edge("apply_belief_inertia", "update_convergence")
    else:
        graph.add_edge("evaluate_hypotheses", "update_convergence")

    graph.add_conditional_edges(
        "update_convergence",
        check_convergence,
        {
            "end": END,
            "loop": "assemble_context",
        },
    )

    return graph.compile()
