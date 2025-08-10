from dotenv import load_dotenv

from .documents.preprocessing import (
    load_markdown_documents,
    split_documents,
)
from .tools.retriever import create_retriever_tool_from_docs
from .graph.workflow import create_graph

# Load environment variables from .env file at the project root
load_dotenv()

# 1. Load and process documents from the local 'datasets' directory
print("--- Loading and processing documents for the graph ---")
docs = load_markdown_documents("datasets")
if not docs:
    raise ValueError("No documents found in the 'datasets' directory. Please add your markdown files.")
doc_splits = split_documents(docs)
print(f"--- {len(docs)} document(s) processed successfully ---\n")

# 2. Create the retriever tool
print("--- Creating retriever tool ---")
retriever_tool = create_retriever_tool_from_docs(doc_splits)
print("--- Retriever tool created successfully ---\n")

# 3. Create the final, servable graph
print("--- Assembling graph ---")
graph = create_graph(retriever_tool)
print("--- Graph assembled and ready to be served ---\n")
