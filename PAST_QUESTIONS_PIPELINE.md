# Past Questions Ingestion Pipeline

## Overview

The past questions pipeline is a separate ingestion flow designed specifically for past exam papers. It differs from the textbook pipeline in the following ways:

### Key Differences

| Aspect | Textbooks | Past Questions |
|--------|-----------|-----------------|
| **Chunking Strategy** | Hierarchical (2-layer) | Flat (single-layer) |
| **Chunk Size** | 2048/512 tokens (parent/child) | 512 tokens (flat) |
| **Purpose** | Context + precision | Question preservation |
| **Image Handling** | Optional | Full support (extracted) |
| **Collection** | `textbook_index` | `past_questions_index` |

## Pipeline Components

### 1. `ingestion/past_questions_loader.py`
- Loads PDFs from a directory
- Extracts both **text** and **images** from PDF pages
- Supports optional metadata mapping from JSON
- Auto-detects year/exam type from filename (e.g., `past_questions_2023_FE.pdf`)

### 2. `ingestion/past_questions_chunker.py`
- Uses flat `SentenceSplitter` (not hierarchical)
- Chunk size: 512 tokens, Overlap: 128 tokens
- Keeps each question/answer intact within a chunk
- Preserves PDF, year, exam type metadata

### 3. `utils/past_questions_mapper.py`
- Parses optional `past_questions_mapping.json`
- Extracts metadata from PDF filenames
- Gracefully falls back if mapping doesn't exist

## Usage

### Method 1: Auto-Detection (No Mapping Required)

If your past questions PDFs are named like `past_questions_<year>_<exam_type>.pdf`:

```bash
python main.py ingest-past-questions --path data/past_questions
```

**Example filenames:**
- `past_questions_2023_FE.pdf`
- `past_questions_2022_FE.pdf`
- `past_questions_2021_AP.pdf`

### Method 2: With Metadata Mapping

If you have a `past_questions_mapping.json`:

```bash
python main.py ingest-past-questions \
  --path data/past_questions \
  --mapping data/past_questions_mapping.json \
  --collection past_exams_2023
```

### Method 3: Multiple Sources

```bash
python main.py ingest-past-questions-multiple \
  --source data/past_questions_2023 \
  --source data/past_questions_2022 \
  --collection past_questions_index
```

## Metadata Schema

Each past question chunk will have these metadata fields in Firestore:

```json
{
  "pdf": "past_questions_2023_FE.pdf",
  "year": 2023,
  "exam_type": "FE",
  "topic": "Hardware",  // from mapping (if provided)
  "difficulty": "medium",  // from mapping (if provided)
  "page": 1,
  "chunk_index": 42,
  "doc_type": "past_question_flat",
  "question_context": "past_exam_scoped"
}
```

## Image Handling

The past questions loader automatically:
1. **Extracts images** from PDF pages
2. **Embeds images** using the same Gemini API
3. **Links images** to their parent document via metadata

Images are stored as separate nodes in the same collection with:
- Image content/URL
- Original page number
- Reference to source PDF

## Optional: Create Mapping File

To create `data/past_questions_mapping.json`:

```json
{
  "exam_sources": [
    {
      "pdf": "past_questions_2023_FE.pdf",
      "questions": [
        {
          "question_id": "Q001",
          "year": 2023,
          "exam_type": "FE",
          "topic": "Hardware",
          "difficulty": "medium"
        }
      ]
    }
  ]
}
```

## Querying Past Questions

When generating new questions, you can query the past questions collection to fetch similar past questions for context:

```python
# Fetch similar past questions for few-shot prompting
past_q_retriever = FirestoreRetriever(
    vectorstore=past_questions_index,
    k=3  # Get top 3 similar past questions
)

# Combined retrieval
textbook_context = textbook_retriever.retrieve("Networks")
past_examples = past_q_retriever.retrieve("Networks")

# Feed both to LLM for generation
```

## Output Structure

```
data/
  past_questions/
    past_questions_2023_FE.pdf
    past_questions_2022_FE.pdf
  past_questions_mapping.example.json  # Reference
```

Firestore Collections:
- `textbook_index` - All hierarchical textbook chunks
- `past_questions_index` - All flat past question chunks
