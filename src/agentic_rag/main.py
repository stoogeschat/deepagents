from dotenv import load_dotenv

from agentic_rag.documents.preprocessing import (
    load_markdown_documents,
    split_documents,
)
from agentic_rag.tools.retriever import create_retriever_tool_from_docs
from agentic_rag.graph.workflow import create_graph

# Load environment variables from .env file
load_dotenv()


def main():
    """
    The main function to run the agentic RAG application.
    """
    # 1. Load and process documents from the local 'datasets' directory
    print("--- Loading and processing documents ---")
    docs = load_markdown_documents("datasets")
    if not docs:
        print("No documents found in the 'datasets' directory. Exiting.")
        return
    doc_splits = split_documents(docs)
    print(f"--- {len(docs)} document(s) processed successfully ---\n")

    # 2. Create the retriever tool
    print("--- Creating retriever tool ---")
    retriever_tool = create_retriever_tool_from_docs(doc_splits)
    print("--- Retriever tool created successfully ---\n")

    # 3. Create the graph
    print("--- Assembling graph ---")
    graph = create_graph(retriever_tool)
    print("--- Graph assembled successfully ---\n")

    # 4. Define a sample question and run the agent
    question = "What are the types of reward hacking?"
    print(f"--- Running agent with question: '{question}' ---\n")

    # 5. Stream the graph execution
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": question}]}
    ):
        for node, update in chunk.items():
            print(f"--- Update from node: {node} ---")
            if "messages" in update:
                last_message = update["messages"][-1]
                if hasattr(last_message, "pretty_print"):
                    last_message.pretty_print()
                else:
                    print(last_message)
            else:
                print(update)
            print("\n")


if __name__ == "__main__":
    main()
