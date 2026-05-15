from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from typing import List


def chunk_documents(documents: List[Document]) -> List[TextNode]:
    """
    Split documents into semantic chunks using SentenceSplitter.

    Args:
        documents: List of Document objects

    Returns:
        List of TextNode chunks with metadata
    """
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separator=" ",
        backup_separators=["\n", ".", ","]
    )

    nodes = splitter.get_nodes_from_documents(documents)

    # Add chunk indexing to metadata
    doc_chunk_count = {}
    for node in nodes:
        source = node.metadata.get("source", "unknown")
        if source not in doc_chunk_count:
            doc_chunk_count[source] = 0

        node.metadata["chunk_index"] = doc_chunk_count[source]
        doc_chunk_count[source] += 1

        # Ensure required fields
        node.metadata.setdefault("doc_type", "concept")
        node.metadata.setdefault("page", 0)
        node.metadata.setdefault("topic", "general")
        node.metadata.setdefault("chapter", None)

        # Clean up chapter_start and chapter_end (internal fields)
        node.metadata.pop("chapter_start", None)
        node.metadata.pop("chapter_end", None)

    return nodes
