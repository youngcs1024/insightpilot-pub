"""Cross-evidence reasoning after the immutable evidence commit barrier."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.degradation import missing_explanations
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.agents.synthesis_generation import generate_synthesis, synthesis_input
from app.core.errors import ConflictError, InsightPilotError


async def synthesize(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Read committed snapshots; never consult history or request new evidence."""
    try:
        ctx = runtime.context
        ctx.deadline.check("synthesize")
        bundle = await ctx.evidence.read_bundle(ctx.identity)
        if bundle.refs != state.evidence_refs:
            raise ConflictError("uncommitted synthesis evidence")
        question = state.rewritten.standalone if state.rewritten else state.question
        result = await generate_synthesis(synthesis_input(question, bundle), bundle, ctx)
        if result.missing_components:
            result = result.model_copy(
                update={
                    "unanswered": list(
                        dict.fromkeys(
                            [*missing_explanations(bundle, state.failures), *result.unanswered]
                        )
                    )[:12]
                }
            )
        return Command(
            update={"synthesis": result, "degraded_components": list(result.missing_components)},
            goto="format_answer",
        )
    except InsightPilotError as exc:
        return failed("synthesize", state, exc)
