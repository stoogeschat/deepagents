from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agentic_rag.core.state import MessagesState
from agentic_rag.graph.nodes import GraphNodes
from agentic_rag.graph.edges import grade_documents


def create_graph(retriever_tool):
    """
    Assembles the agentic RAG graph workflow.

    Args:
        retriever_tool: The retriever tool to be used by the graph.

    Returns:
        A compiled LangGraph.
    """
    # 1. Initialize the node handler class
    nodes = GraphNodes(retriever_tool)

    # 2. Define the graph
    workflow = StateGraph(MessagesState)

    # 3. Add the nodes
    workflow.add_node("generate_query_or_respond", nodes.generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", nodes.rewrite_question)
    workflow.add_node("generate_answer", nodes.generate_answer)

    # 4. Define the edges
    workflow.add_edge(START, "generate_query_or_respond")

    # Conditional edge: Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,  # LangGraph's built-in tool condition
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # Conditional edge: Grade the retrieved documents
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        },
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # 5. Compile the graph
    graph = workflow.compile()
    return graph
