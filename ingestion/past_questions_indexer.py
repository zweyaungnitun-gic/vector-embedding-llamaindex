from typing import List, Dict, Tuple
from llama_index.core.schema import TextNode
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)


def embed_and_store_past_questions(text_nodes: List[TextNode], images_data: List[Dict], collection_name: str = None) -> Tuple[int, int]:
    """
    Generate embeddings for past question text nodes and images using Gemini,
    then store both in Firestore.

    Args:
        text_nodes: List of TextNode objects to embed
        images_data: List of image metadata dicts with image_path
        collection_name: Firestore collection name

    Returns:
        Tuple of (text_nodes_stored, images_stored)
    """
    settings = get_settings()
    if collection_name is None:
        collection_name = "past_questions_index"

    # Initialize Firestore client
    db = settings.get_firestore_client()
    collection_ref = db.collection(collection_name)

    text_stored = 0
    images_stored = 0
    batch_size = 20

    # ====== EMBED AND STORE TEXT NODES ======
    logger.info(f"Processing {len(text_nodes)} text nodes...")
    for i in range(0, len(text_nodes), batch_size):
        batch = text_nodes[i:i + batch_size]
        batch_texts = [node.get_content() for node in batch]

        logger.info(f"Embedding text batch {i // batch_size + 1} ({len(batch)} nodes) using Gemini...")

        try:
            # Generate text embeddings
            embeddings = settings.embed_texts(batch_texts, task_type="RETRIEVAL_DOCUMENT")

            # Store in Firestore
            for node, embedding in zip(batch, embeddings):
                doc_data = {
                    "text": node.get_content(),
                    "embedding": embedding,
                    "metadata": node.metadata,
                    "content_type": "text",
                }
                collection_ref.add(doc_data)
                text_stored += 1

            logger.info(f"Stored {text_stored} text documents so far...")

        except Exception as e:
            logger.error(f"Error embedding text batch: {e}")
            raise

    # ====== EMBED AND STORE IMAGES ======
    if images_data:
        logger.info(f"Processing {len(images_data)} images...")
        
        image_paths = [img["image_path"] for img in images_data]
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_metadata = images_data[i:i + batch_size]

            logger.info(f"Embedding image batch {i // batch_size + 1} ({len(batch_paths)} images) using Gemini multimodal...")

            try:
                # Generate image embeddings using Gemini
                image_embeddings = settings.embed_images(batch_paths)

                # Store in Firestore
                for image_path, metadata, embedding in zip(batch_paths, batch_metadata, image_embeddings):
                    doc_data = {
                        "image_path": image_path,
                        "embedding": embedding,
                        "metadata": metadata,
                        "content_type": "image",
                    }
                    collection_ref.add(doc_data)
                    images_stored += 1

                logger.info(f"Stored {images_stored} image documents so far...")

            except Exception as e:
                logger.error(f"Error embedding image batch: {e}")
                # Continue with next batch instead of failing completely
                logger.warning(f"Skipping image batch due to error: {e}")

    logger.info(f"✓ Successfully stored past questions in Firestore collection: {collection_name}")
    logger.info(f"  - Text nodes: {text_stored}")
    logger.info(f"  - Images: {images_stored}")
    logger.info(f"  - Total: {text_stored + images_stored}")
    
    return text_stored, images_stored
