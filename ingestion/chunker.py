from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from typing import List, Dict
from collections import defaultdict


# Adaptive chunk sizes by topic for optimal question generation
TOPIC_CHUNK_CONFIG = {
    "hardware": {"chunk_size": 768, "chunk_overlap": 192},      # Complex with diagrams
    "network": {"chunk_size": 640, "chunk_overlap": 160},       # Protocols need detail
    "database": {"chunk_size": 640, "chunk_overlap": 160},      # SQL examples need context
    "algorithms": {"chunk_size": 512, "chunk_overlap": 128},    # Code/pseudo-code focused
    "software": {"chunk_size": 512, "chunk_overlap": 128},      # Conceptual
    "systems": {"chunk_size": 640, "chunk_overlap": 160},       # System design
    "security": {"chunk_size": 640, "chunk_overlap": 160},      # Detailed explanations
    "management": {"chunk_size": 500, "chunk_overlap": 125},    # Strategic concepts
    "strategy": {"chunk_size": 500, "chunk_overlap": 125},      # Strategic concepts
    "development": {"chunk_size": 560, "chunk_overlap": 140},   # Development processes
    "legal": {"chunk_size": 480, "chunk_overlap": 120},         # Regulatory details
    "project_management": {"chunk_size": 520, "chunk_overlap": 130},
    "service_management": {"chunk_size": 520, "chunk_overlap": 130},
    "audit": {"chunk_size": 500, "chunk_overlap": 125},
    "systems_strategy": {"chunk_size": 540, "chunk_overlap": 135},
}


def _get_chunk_config(topic: str) -> Dict[str, int]:
    """
    Get chunk configuration for a given topic.
    
    Args:
        topic: Topic name from chapter mapping
        
    Returns:
        Dictionary with chunk_size and chunk_overlap
    """
    return TOPIC_CHUNK_CONFIG.get(topic, {
        "chunk_size": 512,
        "chunk_overlap": 128
    })


def chunk_documents(documents: List[Document]) -> List[TextNode]:
    """
    Split documents into semantic chunks using topic-aware, chapter-bounded approach.
    
    Implements:
    - Adaptive chunk sizes based on topic (hardware chapters get larger chunks)
    - Semantic splitting respecting sentence/paragraph boundaries
    - Chapter-boundary awareness to keep questions focused
    - Enhanced metadata for exam question generation

    Args:
        documents: List of Document objects with chapter metadata

    Returns:
        List of TextNode chunks with enriched metadata for question generation
    """
    nodes = []
    
    # Group documents by chapter to maintain semantic boundaries
    chapters_grouped = defaultdict(list)
    for doc in documents:
        chapter_key = (
            doc.metadata.get("source", "unknown"),
            doc.metadata.get("chapter", "default"),
            doc.metadata.get("topic", "general")
        )
        chapters_grouped[chapter_key].append(doc)
    
    # Track chunk indices per source
    doc_chunk_count = defaultdict(int)
    
    # Process each chapter group separately
    for (source, chapter, topic), chapter_docs in chapters_grouped.items():
        # Get adaptive chunk config for this topic
        config = _get_chunk_config(topic)
        
        splitter = SentenceSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            paragraph_separator="\n\n",
        )
        
        # Split chapter documents
        chapter_nodes = splitter.get_nodes_from_documents(chapter_docs)
        
        # Enrich metadata
        for node in chapter_nodes:
            # Preserve and enhance chapter information
            node.metadata["chapter_name"] = chapter
            node.metadata["topic"] = topic
            node.metadata["source"] = source
            
            # Add chunk index within chapter
            node.metadata["chunk_index"] = doc_chunk_count[source]
            doc_chunk_count[source] += 1
            
            # Add metadata for question generation
            node.metadata["doc_type"] = "concept"
            node.metadata["question_context"] = "chapter_scoped"
            node.metadata["chunk_config"] = f"{config['chunk_size']}/{config['chunk_overlap']}"
            
            # Preserve page information for reference
            if "page" in node.metadata:
                node.metadata["page"] = node.metadata.get("page", 0)
            if "chapter_start" in node.metadata:
                node.metadata["chapter_start_page"] = node.metadata.pop("chapter_start")
            if "chapter_end" in node.metadata:
                node.metadata["chapter_end_page"] = node.metadata.pop("chapter_end")
        
        nodes.extend(chapter_nodes)
    
    return nodes
