import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chapter:
    """Represents a chapter in a textbook."""
    chapter: str
    topic: str
    name: str
    start_page: int
    end_page: int


@dataclass
class TextbookSource:
    """Represents a PDF source with chapters."""
    pdf: str
    chapters: List[Chapter]


def load_chapter_mapping(mapping_file: str) -> Dict[str, TextbookSource]:
    """
    Load chapter mapping from JSON file.

    Args:
        mapping_file: Path to chapter_mapping.json

    Returns:
        Dictionary mapping PDF filenames to TextbookSource objects
    """
    if not Path(mapping_file).exists():
        raise FileNotFoundError(f"Chapter mapping file not found: {mapping_file}")

    with open(mapping_file, 'r') as f:
        data = json.load(f)

    sources = {}
    for textbook in data.get("textbooks", []):
        pdf_name = textbook["pdf"]
        chapters = []
        for key, value in textbook.items():
            if key != "pdf" and isinstance(value, list):
                for ch in value:
                    chapters.append(
                        Chapter(
                            chapter=ch.get("chapter", ""),
                            topic=ch.get("topic", ""),
                            name=ch.get("name", ""),
                            start_page=ch.get("start_page", 0),
                            end_page=ch.get("end_page", 0)
                        )
                    )
        sources[pdf_name] = TextbookSource(pdf=pdf_name, chapters=chapters)

    return sources


def get_chapter_for_page(pdf_name: str, page_num: int, mapping: Dict[str, TextbookSource]) -> Chapter | None:
    """
    Get the chapter that contains a given page number.

    Args:
        pdf_name: PDF filename
        page_num: Page number (1-indexed)
        mapping: Chapter mapping dictionary

    Returns:
        Chapter object or None if page is not in any chapter
    """
    if pdf_name not in mapping:
        return None

    source = mapping[pdf_name]
    for chapter in source.chapters:
        if chapter.start_page <= page_num <= chapter.end_page:
            return chapter

    return None


def validate_mapping(mapping: Dict[str, TextbookSource], pdf_dir: str) -> List[str]:
    """
    Validate that all PDFs in mapping exist.

    Args:
        mapping: Chapter mapping dictionary
        pdf_dir: Directory containing PDFs

    Returns:
        List of errors (empty if valid)
    """
    errors = []
    pdf_path = Path(pdf_dir)

    for pdf_name in mapping.keys():
        if not (pdf_path / pdf_name).exists():
            errors.append(f"PDF not found: {pdf_name}")

    return errors
