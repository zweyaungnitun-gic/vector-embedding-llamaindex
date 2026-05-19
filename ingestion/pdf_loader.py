from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from typing import List, Dict, Optional
from utils.chapter_mapper import load_chapter_mapping, get_chapter_for_page, TextbookSource


def load_textbooks(path: str) -> List[Document]:
    """
    Load PDF textbooks from a directory (without chapter mapping).

    Args:
        path: Directory path containing PDF files

    Returns:
        List of Document objects with metadata
    """
    pdf_dir = Path(path)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    reader = SimpleDirectoryReader(
        input_dir=str(pdf_dir),
        required_exts=[".pdf"],
        recursive=True
    )

    documents = reader.load_data()

    # Tag documents with metadata
    for doc in documents:
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata["doc_type"] = "concept"
        doc.metadata["source"] = doc.metadata.get("file_name", "unknown")

    return documents


def load_textbooks_with_chapters(
    path: str,
    mapping_file: str
) -> List[Document]:
    """
    Load PDF textbooks with chapter information from a mapping file.

    Args:
        path: Directory path containing PDF files
        mapping_file: Path to chapter_mapping.json

    Returns:
        List of Document objects with chapter metadata
    """
    pdf_dir = Path(path)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    # Load chapter mapping
    mapping = load_chapter_mapping(mapping_file)

    # Load all PDFs mentioned in mapping
    documents = []
    for pdf_name, textbook_source in mapping.items():
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = SimpleDirectoryReader(
            input_dir=str(pdf_dir),
            required_exts=[".pdf"]
        )

        # Load the specific PDF
        from llama_index.readers.file import PDFReader
        pdf_reader = PDFReader()
        pdf_docs = pdf_reader.load_data(pdf_path)

        # Tag with chapter information
        for doc in pdf_docs:
            if not doc.metadata:
                doc.metadata = {}

            # Extract page number from metadata
            page_num = doc.metadata.get("page_label")
            if page_num:
                try:
                    page_num = int(page_num)
                except (ValueError, TypeError):
                    page_num = None

            # Find chapter for this page
            if page_num:
                chapter = get_chapter_for_page(pdf_name, page_num, mapping)
                if chapter:
                    doc.metadata["chapter"] = chapter.chapter
                    doc.metadata["topic"] = chapter.topic
                    doc.metadata["name"] = chapter.name
                    doc.metadata["start_page"] = chapter.start_page
                    doc.metadata["end_page"] = chapter.end_page

            doc.metadata["pdf"] = pdf_name
            doc.metadata["doc_type"] = "concept"
            doc.metadata["page"] = page_num or 0

        documents.extend(pdf_docs)

    return documents
