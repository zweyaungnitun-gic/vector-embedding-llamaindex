from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from typing import List, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# Flat chunk sizes for past questions (not hierarchical)
# Each question should ideally fit in one chunk or be clearly separated
# ~256-512 tokens per chunk to keep questions discrete
PAST_QUESTIONS_CHUNK_SIZE = 512
PAST_QUESTIONS_CHUNK_OVERLAP = 128


def chunk_past_questions(documents: List[Document]) -> List[TextNode]:
    """
    Split past exam question documents into flat (non-hierarchical) chunks.
    
    Uses smaller, more conservative chunk sizes to preserve:
    - Question integrity (Q + options + explanation stay together)
    - Page/section boundaries
    - Image-to-text relationships

    Args:
        documents: List of Document objects from past questions PDFs

    Returns:
        List of TextNode chunks optimized for question retrieval
    """
    nodes = []
    
    # Group documents by PDF to maintain question boundaries
    pdfs_grouped = defaultdict(list)
    for doc in documents:
        pdf_name = doc.metadata.get("pdf", "unknown")
        pdfs_grouped[pdf_name].append(doc)
    
    # Track chunk indices per PDF
    doc_chunk_count = defaultdict(int)
    
    # Process each PDF separately
    for pdf_name, pdf_docs in pdfs_grouped.items():
        logger.info(f"Chunking past questions from: {pdf_name}")
        
        # Use flat SentenceSplitter for past questions
        splitter = SentenceSplitter(
            chunk_size=PAST_QUESTIONS_CHUNK_SIZE,
            chunk_overlap=PAST_QUESTIONS_CHUNK_OVERLAP,
            paragraph_separator="\n\n",
            # Use default sentence boundaries
        )
        
        # Split documents
        pdf_nodes = splitter.get_nodes_from_documents(pdf_docs)
        
        # Enrich metadata
        for node in pdf_nodes:
            # Preserve source information
            node.metadata["pdf"] = pdf_name
            node.metadata["year"] = pdf_docs[0].metadata.get("year", 0) if pdf_docs else 0
            node.metadata["exam_type"] = pdf_docs[0].metadata.get("exam_type", "FE") if pdf_docs else "FE"
            
            # Add chunk index within PDF
            node.metadata["chunk_index"] = doc_chunk_count[pdf_name]
            doc_chunk_count[pdf_name] += 1
            
            # Mark as past question chunk
            node.metadata["doc_type"] = "past_question_flat"
            node.metadata["question_context"] = "past_exam_scoped"
            
            # Preserve page information
            if "page" in node.metadata:
                node.metadata["page"] = node.metadata.get("page", 0)
            
            logger.debug(f"Created chunk {node.metadata['chunk_index']} from {pdf_name}")
        
        nodes.extend(pdf_nodes)
    
    logger.info(f"✓ Created {len(nodes)} flat chunks from past questions")
    return nodes
