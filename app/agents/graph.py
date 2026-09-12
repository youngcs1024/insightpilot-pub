"""Parent topology with an isolated data specialist; production compilation requires a PostgreSQL saver."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.answer_data import answer_data_node
from app.agents.nodes.format_answer import format_answer
from app.agents.nodes.persist_evidence import persist_evidence
from app.agents.nodes.prepare import prepare
from app.agents.nodes.rewrite_question import rewrite_question
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState, GraphInput, GraphOutput

RECURSION_LIMIT = 32
type PhaseOneGraph = CompiledStateGraph[AgentState, RuntimeContext, GraphInput, GraphOutput]


def topology() -> StateGraph[AgentState, RuntimeContext, GraphInput, GraphOutput]:
    """Build the topology without opening a connection or choosing volatile storage."""
    graph = StateGraph(
        AgentState,
        context_schema=RuntimeContext,
        input_schema=GraphInput,
        output_schema=GraphOutput,
    )
    graph.add_node("prepare", prepare, destinations=("rewrite_question", "__end__"))
    graph.add_node("rewrite_question", rewrite_question, destinations=("answer_data", "__end__"))
    graph.add_node("answer_data", answer_data_node, destinations=("persist_evidence", "__end__"))
    graph.add_node("persist_evidence", persist_evidence, destinations=("format_answer", "__end__"))
    graph.add_node("format_answer", format_answer, destinations=("__end__",))
    graph.add_edge(START, "prepare")
    return graph


def build(checkpointer: AsyncPostgresSaver) -> PhaseOneGraph:
    """There is deliberately no in-memory or checkpoint-free production fallback."""
    return topology().compile(checkpointer=checkpointer, name="insightpilot")


if __name__ == "__main__":
    # Inspection only: production build() always requires a PostgreSQL saver.
    for edge in topology().compile(name="insightpilot-diagram").get_graph().edges:
        print(f"{edge.source} -> {edge.target}")
