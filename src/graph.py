from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from src.state import AxiomCartState
from src.config import get_logger
from src.nodes import orchestrator_node, product_agent, support_agent, synthesizer_node


logger = get_logger("graph")

def build_graph() -> StateGraph:
    """create wire and compile the AxiomCart multi-agent graph"""

    builder = StateGraph(AxiomCartState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("product_agent", product_agent)
    builder.add_node("support_agent", support_agent)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "orchestrator")
    builder.add_edge("synthesizer", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    logger.info("Graph compiled (with memory saver for conversation persistence)")
    return graph

axiom_cart_graph = build_graph()


