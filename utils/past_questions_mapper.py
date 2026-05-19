import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PastQuestion:
    """Represents a past exam question with metadata."""
    question_id: str
    year: int
    exam_type: str
    topic: str
    difficulty: str
    session: str = ""


@dataclass
class PastQuestionsSource:
    """Represents a past questions PDF source."""
    pdf: str
    questions: List[PastQuestion]


def load_past_questions_mapping(mapping_file: str = None) -> Dict[str, PastQuestionsSource]:
    """
    Load past questions mapping from JSON file.
    If no mapping file exists, returns empty dict (auto-detection mode).

    Args:
        mapping_file: Path to past_questions_mapping.json (optional)

    Returns:
        Dictionary mapping PDF filenames to PastQuestionsSource objects
    """
    if mapping_file is None:
        # Auto-detection mode - return empty dict, metadata will be inferred from filenames
        return {}

    if not Path(mapping_file).exists():
        # File doesn't exist, return empty dict for graceful fallback
        return {}

    with open(mapping_file, 'r') as f:
        data = json.load(f)

    sources = {}
    for exam_source in data.get("exam_sources", []):
        pdf_name = exam_source["pdf"]
        questions = [
            PastQuestion(
                question_id=q.get("question_id", ""),
                year=q.get("year", 0),
                exam_type=q.get("exam_type", "FE"),
                topic=q.get("topic", "general"),
                difficulty=q.get("difficulty", "medium"),
                session=q.get("session", "")
            )
            for q in exam_source.get("questions", [])
        ]
        sources[pdf_name] = PastQuestionsSource(pdf=pdf_name, questions=questions)

    return sources


def get_question_metadata_from_filename(pdf_name: str) -> Dict[str, Any]:
    """
    Extract metadata from past questions PDF filename.
    
    Expected format: past_questions_<year>_<exam_type>.pdf
    Example: past_questions_2023_FE.pdf

    Args:
        pdf_name: PDF filename

    Returns:
        Dictionary with inferred metadata
    """
    metadata = {
        "year": 0,
        "exam_type": "FE",
        "source": pdf_name,
        "session": ""
    }

    # Try to extract year and exam type from filename
    parts = pdf_name.replace(".pdf", "").split("_")
    try:
        if len(parts) >= 3:
            if parts[1].isdigit():
                metadata["year"] = int(parts[1])
            if len(parts) >= 3:
                metadata["exam_type"] = parts[2].upper()
    except (ValueError, IndexError):
        pass

    return metadata
