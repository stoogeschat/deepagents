import getpass
import os

from agentic_rag.config import DOCUMENT_URLS
from agentic_rag.documents.preprocessing import load_documents, split_documents
from agentic_rag.tools.retriever import create_retriever_tool_from_docs
from agentic_rag.graph.workflow import create_graph


def _set_env(key: str):
    """
    Set an environment variable if it is not already set.
    Prompts the user for the value.
    """
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"Please enter your {key}: ")


def main():
    """
    The main function to run the agentic RAG application.
    """
    # 1. Set the OpenAI API key
    _set_env("OPENAI_API_KEY")

    # 2. Load and process documents
    print("--- Loading and processing documents ---")
    docs = load_documents(DOCUMENT_URLS)
    doc_splits = split_documents(docs)
    print("--- Documents processed successfully ---\n")

    # 3. Create the retriever tool
    print("--- Creating retriever tool ---")
    retriever_tool = create_retriever_tool_from_docs(doc_splits)
    print("--- Retriever tool created successfully ---\n")

    # 4. Create the graph
    print("--- Assembling graph ---")
    graph = create_graph(retriever_tool)
    print("--- Graph assembled successfully ---\n")

    # 5. Define a sample question and run the agent
    question = "What does Lilian Weng say about types of reward hacking?"
    print(f"--- Running agent with question: '{question}' ---\n")

    # 6. Stream the graph execution
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": question}]}
    ):
        for node, update in chunk.items():
            print(f"--- Update from node: {node} ---")
            if "messages" in update:
                # Get the last message and pretty print it
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
