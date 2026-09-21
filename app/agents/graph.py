"""Four bounded routes, with a same-super-step specialist completion barrier."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.contracts import Route
from app.agents.nodes.answer_data import answer_data_routed_node
from app.agents.nodes.answer_knowledge import answer_knowledge_node
from app.agents.nodes.clarify import clarify
from app.agents.nodes.finalize_context import finalize_context
from app.agents.nodes.format_answer import format_answer
from app.agents.nodes.persist_evidence import persist_evidence
from app.agents.nodes.prepare import prepare
from app.agents.nodes.router import parent_router
from app.agents.nodes.synthesis import synthesize
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState, GraphInput, GraphOutput

RECURSION_LIMIT = 32
type PhaseOneGraph = CompiledStateGraph[AgentState, RuntimeContext, GraphInput, GraphOutput]


def dispatch(state: AgentState) -> list[str] | str:
    """Failures before dispatch must never schedule an arbitrary specialist."""
    if state.failures or state.route is None:
        return "__end__"
    match state.route.route:
        case Route.DATA_ONLY:
            return "data_agent"
        case Route.KNOWLEDGE_ONLY:
            return "knowledge_agent"
        case Route.BOTH:
            return ["data_agent", "knowledge_agent"]
        case Route.CLARIFY:
            return "clarify"


def topology() -> StateGraph[AgentState, RuntimeContext, GraphInput, GraphOutput]:
    """Use static edges only for update-only nodes; Commands own terminal routing."""
    graph = StateGraph(
        AgentState,
        context_schema=RuntimeContext,
        input_schema=GraphInput,
        output_schema=GraphOutput,
    )
    graph.add_node("prepare_context", prepare, destinations=("route", "__end__"))
    graph.add_node("route", parent_router, destinations=("finalize_context", "__end__"))
    graph.add_node("finalize_context", finalize_context)
    graph.add_node("data_agent", answer_data_routed_node)
    graph.add_node("knowledge_agent", answer_knowledge_node)
    graph.add_node("clarify", clarify)
    graph.add_node(
        "persist_evidence",
        persist_evidence,
        destinations=("synthesize", "format_answer", "__end__"),
    )
    graph.add_node("synthesize", synthesize, destinations=("format_answer", "__end__"))
    graph.add_node("format_answer", format_answer, destinations=("__end__",))
    graph.add_edge(START, "prepare_context")
    graph.add_conditional_edges(
        "finalize_context",
        dispatch,
        path_map=["data_agent", "knowledge_agent", "clarify", "__end__"],
    )
    graph.add_edge("data_agent", "persist_evidence")
    graph.add_edge("knowledge_agent", "persist_evidence")
    graph.add_edge("clarify", "format_answer")
    return graph


def build(checkpointer: AsyncPostgresSaver) -> PhaseOneGraph:
    """Production requires the application-owned PostgreSQL saver."""
    return topology().compile(checkpointer=checkpointer, name="insightpilot")


if __name__ == "__main__":
    for edge in topology().compile(name="insightpilot-diagram").get_graph().edges:
        print(f"{edge.source} -> {edge.target}")
