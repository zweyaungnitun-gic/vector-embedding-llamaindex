from typing import List
from llama_index.core.schema import TextNode
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)


def embed_and_store(nodes: List[TextNode], collection_name: str = None) -> int:
    """
    Generate embeddings for text nodes using Gemini text-embedding-004 (via google-genai SDK)
    and store in Firestore.

    Args:
        nodes: List of TextNode objects to embed
        collection_name: Firestore collection name

    Returns:
        Number of documents stored
    """
    settings = get_settings()
    if collection_name is None:
        collection_name = settings.firestore_collection

    # Initialize Firestore client
    db = settings.get_firestore_client()
    collection_ref = db.collection(collection_name)

    total_stored = 0
    batch_size = 20

    # Process nodes in batches
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        batch_texts = [node.get_content() for node in batch]

        logger.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} nodes) using Gemini text-embedding-004...")

        try:
            # Generate embeddings using Gemini text-embedding-004 (v1 stable API)
            embeddings = settings.embed_texts(batch_texts, task_type="RETRIEVAL_DOCUMENT")

            # Store in Firestore
            for node, embedding in zip(batch, embeddings):
                doc_data = {
                    "text": node.get_content(),
                    "embedding": embedding,
                    "metadata": node.metadata,
                }

                # Use auto-generated document ID
                collection_ref.add(doc_data)
                total_stored += 1

            logger.info(f"Stored {total_stored} documents so far...")

        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise

    logger.info(f"✓ Successfully stored {total_stored} documents in Firestore collection: {collection_name}")
    return total_stored
