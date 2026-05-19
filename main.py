import click
import logging
from pathlib import Path
from ingestion.pdf_loader import load_textbooks, load_textbooks_with_chapters
from ingestion.chunker import chunk_documents
from ingestion.past_questions_loader import load_past_questions
from ingestion.past_questions_chunker import chunk_past_questions
from ingestion.past_questions_indexer import embed_and_store_past_questions
from ingestion.indexer import embed_and_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """ITPEC FE RAG Vector Embedding Pipeline"""
    pass


@cli.command()
@click.option(
    '--path',
    required=True,
    type=click.Path(exists=True),
    help='Path to directory containing PDF textbooks'
)
@click.option(
    '--mapping',
    default=None,
    type=click.Path(exists=True),
    help='Path to chapter mapping JSON file'
)
@click.option(
    '--collection',
    default=None,
    help='Firestore collection name (defaults to fe_vector_index)'
)
def ingest(path: str, mapping: str, collection: str):
    """Ingest PDF textbooks, chunk, embed, and store in Firestore."""
    try:
        logger.info(f"Starting ingestion pipeline...")
        logger.info(f"Loading PDFs from: {path}")

        # Step 1: Load PDFs
        if mapping:
            logger.info(f"Using chapter mapping from: {mapping}")
            documents = load_textbooks_with_chapters(path, mapping)
        else:
            documents = load_textbooks(path)
        logger.info(f"✓ Loaded {len(documents)} documents")

        # Step 2: Chunk documents
        logger.info(f"Chunking documents...")
        nodes = chunk_documents(documents)
        logger.info(f"✓ Created {len(nodes)} chunks")

        # Step 3: Embed and store
        logger.info(f"Embedding and storing in Firestore...")
        stored_count = embed_and_store(nodes, collection_name=collection)

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Pipeline complete!")
        logger.info(f"  - Documents loaded: {len(documents)}")
        logger.info(f"  - Chunks created: {len(nodes)}")
        logger.info(f"  - Stored in Firestore: {stored_count}")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise


@cli.command()
@click.option(
    '--source',
    multiple=True,
    type=click.Path(exists=True),
    required=True,
    help='Path to PDF source (can specify multiple times)'
)
@click.option(
    '--mapping',
    default=None,
    type=click.Path(exists=True),
    help='Path to chapter mapping JSON file'
)
@click.option(
    '--collection',
    default=None,
    help='Firestore collection name'
)
def ingest_multiple(source: tuple, mapping: str, collection: str):
    """Ingest from multiple PDF sources."""
    try:
        all_documents = []
        all_nodes = []

        for source_path in source:
            logger.info(f"Processing: {source_path}")
            if mapping:
                documents = load_textbooks_with_chapters(source_path, mapping)
            else:
                documents = load_textbooks(source_path)
            all_documents.extend(documents)
            logger.info(f"  ✓ Loaded {len(documents)} documents")

        logger.info(f"Chunking {len(all_documents)} total documents...")
        all_nodes = chunk_documents(all_documents)
        logger.info(f"✓ Created {len(all_nodes)} chunks")

        logger.info(f"Embedding and storing...")
        stored_count = embed_and_store(all_nodes, collection_name=collection)

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Multi-source ingestion complete!")
        logger.info(f"  - Total documents: {len(all_documents)}")
        logger.info(f"  - Total chunks: {len(all_nodes)}")
        logger.info(f"  - Stored in Firestore: {stored_count}")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error during multi-source ingestion: {e}")
        raise


@cli.command()
@click.option(
    '--path',
    required=True,
    type=click.Path(exists=True),
    help='Path to directory containing past questions PDFs'
)
@click.option(
    '--mapping',
    default=None,
    type=click.Path(exists=True),
    help='Path to past questions mapping JSON file (optional)'
)
@click.option(
    '--collection',
    default=None,
    help='Firestore collection name (defaults to past_questions_index)'
)
def ingest_past_questions(path: str, mapping: str, collection: str):
    """Ingest past exam questions PDFs with text and image extraction, chunk, embed, and store in Firestore."""
    try:
        logger.info(f"Starting past questions ingestion pipeline...")
        logger.info(f"Loading past question PDFs from: {path}")

        # Step 1: Load PDFs (with image support)
        if mapping:
            logger.info(f"Using past questions mapping from: {mapping}")
        text_documents, images_data = load_past_questions(path, mapping_file=mapping)
        logger.info(f"✓ Loaded {len(text_documents)} text segments and {len(images_data)} images")

        # Step 2: Chunk text documents (flat, non-hierarchical)
        logger.info(f"Chunking past questions...")
        text_nodes = chunk_past_questions(text_documents)
        logger.info(f"✓ Created {len(text_nodes)} text chunks")

        # Step 3: Embed and store (text + images)
        if collection is None:
            collection = "past_questions_index"
        logger.info(f"Embedding and storing in Firestore collection: {collection}...")
        text_stored, images_stored = embed_and_store_past_questions(text_nodes, images_data, collection_name=collection)

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Past questions ingestion complete!")
        logger.info(f"  - Text segments loaded: {len(text_documents)}")
        logger.info(f"  - Text chunks created: {len(text_nodes)}")
        logger.info(f"  - Text chunks stored: {text_stored}")
        logger.info(f"  - Images extracted: {len(images_data)}")
        logger.info(f"  - Images stored: {images_stored}")
        logger.info(f"  - Collection: {collection}")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error during past questions ingestion: {e}")
        raise


@cli.command()
@click.option(
    '--source',
    multiple=True,
    type=click.Path(exists=True),
    required=True,
    help='Path to past questions PDF source (can specify multiple times)'
)
@click.option(
    '--mapping',
    default=None,
    type=click.Path(exists=True),
    help='Path to past questions mapping JSON file (optional)'
)
@click.option(
    '--collection',
    default=None,
    help='Firestore collection name'
)
def ingest_past_questions_multiple(source: tuple, mapping: str, collection: str):
    """Ingest past questions from multiple PDF sources."""
    try:
        all_text_documents = []
        all_images_data = []
        all_text_nodes = []

        for source_path in source:
            logger.info(f"Processing: {source_path}")
            text_documents, images_data = load_past_questions(source_path, mapping_file=mapping)
            all_text_documents.extend(text_documents)
            all_images_data.extend(images_data)
            logger.info(f"  ✓ Loaded {len(text_documents)} text segments and {len(images_data)} images")

        logger.info(f"Chunking {len(all_text_documents)} total text documents...")
        all_text_nodes = chunk_past_questions(all_text_documents)
        logger.info(f"✓ Created {len(all_text_nodes)} text chunks")

        if collection is None:
            collection = "past_questions_index"
        logger.info(f"Embedding and storing in collection: {collection}...")
        text_stored, images_stored = embed_and_store_past_questions(all_text_nodes, all_images_data, collection_name=collection)

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Multi-source past questions ingestion complete!")
        logger.info(f"  - Total text documents: {len(all_text_documents)}")
        logger.info(f"  - Total text chunks: {len(all_text_nodes)}")
        logger.info(f"  - Text chunks stored: {text_stored}")
        logger.info(f"  - Total images: {len(all_images_data)}")
        logger.info(f"  - Images stored: {images_stored}")
        logger.info(f"  - Collection: {collection}")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error during multi-source past questions ingestion: {e}")
        raise


if __name__ == "__main__":
    cli()
