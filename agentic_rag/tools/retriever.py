from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from agentic_rag.config import (
    EMBEDDING_MODEL_NAME,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
)


def create_retriever_tool_from_docs(docs: list):
    """
    Create a retriever tool from a list of documents using a local embedding model.

    Args:
        docs: A list of document chunks to be indexed.

    Returns:
        A LangChain retriever tool.
    """
    # 1. Initialize embeddings with local model configuration
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
    )

    # 2. Create an in-memory vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=docs, embedding=embeddings
    )

    # 3. Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    # 4. Create a retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts.",
    )

    return retriever_tool
