from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.schema import Document, TextNode
from typing import List, Dict
from collections import defaultdict


# Topic agnostic hierarchical chunk sizes
# Parent chunks -> ~2048 tokens (1-3 pages, concept level)
# Child chunks -> ~512 tokens (detail level, paragraphs)
HIERARCHICAL_CHUNK_SIZES = [2048, 512]


def chunk_documents(documents: List[Document]) -> List[TextNode]:
    """
    Split documents into semantic hierarchical chunks.
    
    Implements 2-layer indexing (Option C) for better question generation:
    - Parent chunks: Concept level (1-3 pages per node) for context
    - Child chunks: Detail level (paragraphs/sentences) for precision
    """
    nodes = []
    
    # Group documents by chapter to maintain semantic boundaries
    chapters_grouped = defaultdict(list)
    for doc in documents:
        chapter_key = (
            doc.metadata.get("pdf", doc.metadata.get("source", "unknown")),
            doc.metadata.get("chapter", "default"),
            doc.metadata.get("topic", "general"),
            doc.metadata.get("name", "general")
        )
        chapters_grouped[chapter_key].append(doc)
    
    # Track chunk indices per source
    doc_chunk_count = defaultdict(int)
    
    # Process each chapter group separately
    for (source, chapter, topic, name), chapter_docs in chapters_grouped.items():
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=HIERARCHICAL_CHUNK_SIZES
        )
        
        # Split chapter documents into hierarchical root nodes
        root_nodes = node_parser.get_nodes_from_documents(chapter_docs)
        
        # Extract leaf nodes
        leaf_nodes = get_leaf_nodes(root_nodes)
        
        # Combine all nodes for indexing in Firestore
        all_nodes = root_nodes + leaf_nodes
        
        # Enrich metadata
        for node in all_nodes:
            # Match chapter_mapping.json fields directly
            node.metadata["chapter"] = chapter
            node.metadata["topic"] = topic
            node.metadata["name"] = name
            node.metadata["pdf"] = source
            
            # Add chunk index within chapter
            node.metadata["chunk_index"] = doc_chunk_count[source]
            doc_chunk_count[source] += 1
            
            # Add metadata for question generation
            node.metadata["doc_type"] = "concept_hierarchy"
            node.metadata["question_context"] = "hierarchical_scoped"
            
            # Preserve page information for reference
            if "page" in node.metadata:
                node.metadata["page"] = node.metadata.get("page", 0)
            if "start_page" in node.metadata:
                pass # Already correct
            elif "chapter_start" in node.metadata:
                node.metadata["start_page"] = node.metadata.pop("chapter_start")
            
            if "end_page" in node.metadata:
                pass # Already correct
            elif "chapter_end" in node.metadata:
                node.metadata["end_page"] = node.metadata.pop("chapter_end")
        
        nodes.extend(all_nodes)
    
    return nodes
