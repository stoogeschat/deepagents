from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(urls: list[str]) -> list:
    """
    Load documents from a list of URLs.

    Args:
        urls: A list of URLs to load documents from.

    Returns:
        A list of loaded documents.
    """
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]


def split_documents(docs: list) -> list:
    """
    Split documents into smaller chunks.

    Args:
        docs: A list of documents to split.

    Returns:
        A list of smaller document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    return text_splitter.split_documents(docs)
