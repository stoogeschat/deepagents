from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_markdown_documents(directory_path: str) -> list:
    """
    Load Markdown documents from a specified directory.

    Args:
        directory_path: The path to the directory containing Markdown files.

    Returns:
        A list of loaded documents.
    """
    # Use DirectoryLoader to load all .md files
    loader = DirectoryLoader(directory_path, glob="**/*.md", show_progress=True)
    docs = loader.load()
    return docs


def split_documents(docs: list) -> list:
    """
    Split documents into smaller chunks.

    Args:
        docs: A list of documents to split.

    Returns:
        A list of smaller document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,  # Increased chunk size for potentially larger markdown docs
        chunk_overlap=100,
    )
    return text_splitter.split_documents(docs)
