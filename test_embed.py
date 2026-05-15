#!/usr/bin/env python3
"""
Test script to verify Vertex AI embedding + Firestore storage pipeline.
Creates 5 sample TextNodes with FE exam topics, embeds them, and stores in Firestore.
Then reads back and verifies.
"""

import logging
from llama_index.core.schema import TextNode
from config.settings import get_settings
from ingestion.indexer import embed_and_store

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample FE exam topic content
SAMPLE_NODES_DATA = [
    {
        "topic": "algorithms",
        "chapter": "Chapter 1: Algorithms and Computational Complexity",
        "text": "Binary search is an efficient algorithm for finding a target value in a sorted array. "
                "It works by repeatedly dividing the search interval in half. Starting with the entire array, "
                "it compares the target with the middle element. If the target matches, the position is returned. "
                "If the target is less, the search continues in the left half; otherwise in the right half. "
                "Time complexity: O(log n). Space complexity: O(1) for iterative, O(log n) for recursive due to call stack.",
    },
    {
        "topic": "data_structures",
        "chapter": "Chapter 2: Data Structures",
        "text": "Stacks and queues are fundamental linear data structures with different access patterns. "
                "A stack (LIFO - Last In First Out) uses push to add elements and pop to remove from the same end. "
                "A queue (FIFO - First In First Out) uses enqueue to add at the rear and dequeue to remove from the front. "
                "Stacks are used in function call management, undo operations, and expression parsing. "
                "Queues are used in breadth-first search, task scheduling, and message passing systems.",
    },
    {
        "topic": "networks",
        "chapter": "Chapter 5: Networks and Protocols",
        "text": "The TCP three-way handshake establishes a reliable connection between client and server. "
                "Step 1 (SYN): Client sends SYN packet with sequence number to server. "
                "Step 2 (SYN-ACK): Server responds with SYN-ACK, acknowledging the client's sequence and sending its own. "
                "Step 3 (ACK): Client sends ACK, confirming the server's sequence number. "
                "This process ensures both sides are ready for data transmission and synchronize sequence numbers for reliability. "
                "Once established, data flows reliably with checksums and retransmission on errors.",
    },
    {
        "topic": "databases",
        "chapter": "Chapter 6: Databases",
        "text": "ACID properties ensure database transactions are reliable and consistent. "
                "Atomicity: A transaction is all-or-nothing; either all changes commit or all rollback. "
                "Consistency: The database moves from one valid state to another; all constraints are maintained. "
                "Isolation: Concurrent transactions do not interfere with each other; each executes as if alone. "
                "Durability: Once committed, changes persist even after system failures. "
                "Modern databases use write-ahead logging, locking mechanisms, and snapshot isolation to guarantee ACID properties.",
    },
    {
        "topic": "security",
        "chapter": "Chapter 8: Security",
        "text": "Symmetric and asymmetric encryption serve different purposes in secure communication. "
                "Symmetric encryption (AES, DES) uses a single shared key for both encryption and decryption. "
                "It is fast and efficient for large data but requires secure key exchange beforehand. "
                "Asymmetric encryption (RSA, ECC) uses a public-private key pair; public key encrypts, private key decrypts. "
                "It solves the key distribution problem but is computationally expensive. "
                "Hybrid approaches use asymmetric encryption to exchange a symmetric key, combining both advantages.",
    },
]


def create_sample_nodes() -> list:
    """Create 5 sample TextNode objects for testing."""
    nodes = []
    for i, data in enumerate(SAMPLE_NODES_DATA):
        node = TextNode(
            text=data["text"],
            metadata={
                "doc_type": "concept",
                "chapter": data["chapter"],
                "topic": data["topic"],
                "page": 0,
                "chunk_index": i,
                "source": "test_sample.txt",
            }
        )
        nodes.append(node)
    return nodes


def verify_firestore_data(test_collection: str = "fe_vector_test") -> bool:
    """Read back stored documents from Firestore and verify."""
    settings = get_settings()
    db = settings.get_firestore_client()
    collection_ref = db.collection(test_collection)

    docs = collection_ref.stream()
    doc_list = list(docs)

    if len(doc_list) != 5:
        logger.error(f"Expected 5 documents, found {len(doc_list)}")
        return False

    logger.info(f"\nVerification Results:")
    logger.info(f"{'='*70}")

    for idx, doc in enumerate(doc_list, 1):
        data = doc.to_dict()

        # Check required fields
        if not all(k in data for k in ["text", "embedding", "metadata"]):
            logger.error(f"  [{idx}] Missing required fields")
            return False

        embedding = data.get("embedding", [])
        metadata = data.get("metadata", {})
        topic = metadata.get("topic", "unknown")
        text_preview = data.get("text", "")[:50] + "..."

        # Verify embedding dimensions
        embedding_dims = len(embedding) if isinstance(embedding, list) else 0
        if embedding_dims != 768:
            logger.error(f"  [{idx}] Expected 768-dim embedding, got {embedding_dims}")
            return False

        logger.info(f"  [{idx}] ✓ topic={topic:20s} | embedding_dims={embedding_dims} | text={text_preview}")

    logger.info(f"{'='*70}")
    return True


def main():
    """Run the embedding test."""
    test_collection = "fe_vector_test"

    logger.info(f"\n{'='*70}")
    logger.info(f"Starting embedding test with sample data...")
    logger.info(f"{'='*70}\n")

    try:
        # Step 1: Create sample nodes
        logger.info("Step 1: Creating 5 sample TextNode objects...")
        nodes = create_sample_nodes()
        logger.info(f"✓ Created {len(nodes)} sample nodes\n")

        # Step 2: Embed and store
        logger.info(f"Step 2: Embedding and storing in Firestore collection '{test_collection}'...")
        stored_count = embed_and_store(nodes, collection_name=test_collection)
        logger.info(f"✓ Stored {stored_count} documents\n")

        # Step 3: Verify
        logger.info(f"Step 3: Verifying data in Firestore...")
        if verify_firestore_data(test_collection):
            logger.info(f"\n{'='*70}")
            logger.info(f"✓ TEST PASSED: All embeddings stored and verified!")
            logger.info(f"{'='*70}\n")
            return True
        else:
            logger.error(f"\n{'='*70}")
            logger.error(f"✗ TEST FAILED: Verification error")
            logger.error(f"{'='*70}\n")
            return False

    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"✗ TEST FAILED: {e}")
        logger.error(f"{'='*70}\n")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
