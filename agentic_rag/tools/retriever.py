from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


def create_retriever_tool_from_docs(docs: list):
    """
    Create a retriever tool from a list of documents.

    Args:
        docs: A list of document chunks to be indexed.

    Returns:
        A LangChain retriever tool.
    """
    # 1. Create an in-memory vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=docs, embedding=OpenAIEmbeddings()
    )

    # 2. Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    # 3. Create a retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts.",
    )

    return retriever_tool
