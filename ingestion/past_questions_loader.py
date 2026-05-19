from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document, ImageDocument
from typing import List, Dict, Optional, Tuple
from utils.past_questions_mapper import load_past_questions_mapping, get_question_metadata_from_filename
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


def extract_images_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extract images from PDF and save to temporary directory.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of tuples (image_path, page_number)
    """
    # Prefer PyMuPDF (fitz) to avoid external Poppler dependency required by pdf2image.
    try:
        import fitz  # PyMuPDF
    except Exception:
        logger.warning("PyMuPDF (fitz) not installed; skipping PDF image extraction.")
        return []

    try:
        temp_dir = tempfile.mkdtemp()
        doc = fitz.open(pdf_path)
        image_paths = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            image_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
            pix.save(image_path)
            image_paths.append((image_path, page_num + 1))

        logger.info(f"Extracted {len(image_paths)} images from {Path(pdf_path).name} using PyMuPDF")
        return image_paths
    except Exception as e:
        logger.warning(f"Could not extract images from PDF using PyMuPDF: {e}")
        return []


def load_past_questions(path: str, mapping_file: str = None) -> Tuple[List[Document], List[Dict]]:
    """
    Load past exam questions PDFs with both text and image extraction.

    Args:
        path: Directory path containing past questions PDFs
        mapping_file: Optional path to past_questions_mapping.json

    Returns:
        Tuple of (List of Document objects with text, List of image data dicts)
    """
    pdf_dir = Path(path)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    # Load mapping if provided
    mapping = load_past_questions_mapping(mapping_file) if mapping_file else {}

    documents = []
    images_data = []  # Store image metadata and paths
    
    # Load PDFs
    from llama_index.readers.file import PDFReader
    
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    for pdf_path in pdf_files:
        pdf_name = pdf_path.name
        logger.info(f"Loading past questions from: {pdf_name}")
        
        # Load text with PDFReader
        pdf_reader = PDFReader()
        pdf_docs = pdf_reader.load_data(pdf_path)
        
        # Get metadata from filename or mapping
        if pdf_name in mapping:
            question_source = mapping[pdf_name]
            source_metadata = {
                "pdf": pdf_name,
                "year": question_source.questions[0].year if question_source.questions else 0,
                "exam_type": question_source.questions[0].exam_type if question_source.questions else "FE",
                "session": question_source.questions[0].session if question_source.questions else ""
            }
        else:
            source_metadata = get_question_metadata_from_filename(pdf_name)
        
        # Tag text documents
        for doc in pdf_docs:
            if not doc.metadata:
                doc.metadata = {}
            
            # Add source metadata
            doc.metadata.update(source_metadata)
            doc.metadata["doc_type"] = "past_question"
            
            # Extract page number
            page_num = doc.metadata.get("page_label")
            if page_num:
                try:
                    doc.metadata["page"] = int(page_num)
                except (ValueError, TypeError):
                    doc.metadata["page"] = 0
            
            logger.debug(f"Tagged text document from {pdf_name}, page {doc.metadata.get('page', 'unknown')}")
        
        documents.extend(pdf_docs)
        
        # Extract images from PDF
        image_paths = extract_images_from_pdf(str(pdf_path))
        for image_path, page_num in image_paths:
            image_metadata = {
                "pdf": pdf_name,
                "page": page_num,
                "image_path": image_path,
                "doc_type": "past_question_image",
            }
            image_metadata.update(source_metadata)
            images_data.append(image_metadata)
    
    logger.info(f"✓ Loaded {len(documents)} text segments from past questions")
    logger.info(f"✓ Extracted {len(images_data)} images from past questions")
    return documents, images_data
