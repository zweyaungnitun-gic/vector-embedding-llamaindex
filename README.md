# ITPEC FE Exam Question Generation with RAG

A Python-based RAG (Retrieval-Augmented Generation) system for generating ITPEC Fundamental Engineer exam questions using vector embeddings stored in Firebase Firestore.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Credentials

```bash
# Copy environment template
cp .env.example .env

# Edit .env and fill in:
# - GOOGLE_API_KEY: Get from https://makersuite.google.com/app/apikey
# - GOOGLE_APPLICATION_CREDENTIALS: Path to Firebase service account JSON
# - Place your firebase-credentials.json in repo root
```

### 3. Add PDF Textbooks

Place your PDF files in:
- `data/textbooks/` — FE textbooks
- `data/past_questions/` — Past exam papers

### 4. Set Up Chapter Mapping (Optional)

Edit `data/chapter_mapping.json` to map PDF files to chapters:

```json
{
  "textbooks": [
    {
      "pdf": "textbook_1.pdf",
      "chapters": [
        {
          "name": "Chapter 1: Algorithms",
          "start_page": 1,
          "end_page": 45,
          "topic": "algorithms"
        }
      ]
    }
  ]
}
```

### 5. Run Ingestion Pipeline

```bash
# Ingest without chapter mapping
python main.py ingest --path data/textbooks/

# Ingest with chapter mapping (recommended)
python main.py ingest --path data/textbooks/ --mapping data/chapter_mapping.json

# Ingest multiple sources at once
python main.py ingest-multiple --source data/textbooks/ --source data/past_questions/ --mapping data/chapter_mapping.json
```

This will:
1. Load PDFs from the directory
2. Tag chunks with chapter information (if mapping provided)
3. Split into **adaptive semantic chunks** (480-768 tokens by topic, 120-192 overlap)
   - Chapter-bounded: Chunks respect chapter boundaries for focused questions
   - Topic-aware: Hardware/Networks get larger chunks (768t), Algorithms/Management get smaller (480-512t)
4. Generate embeddings using Vertex AI text-embedding-004 (768-dim)
5. Store in Firestore collection `fe_vector_index` with enhanced metadata

## Project Structure

```
vector-embedding-llamaindex/
├── config/
│   └── settings.py          # Firebase & Vertex AI initialization
├── ingestion/
│   ├── pdf_loader.py        # Load PDFs
│   ├── chunker.py           # Split into chunks
│   └── indexer.py           # Embed & store in Firestore
├── data/
│   ├── textbooks/           # Place FE textbooks here
│   └── past_questions/      # Place past exams here
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── ARCHITECTURE.md          # Detailed architecture documentation
```

## Pipeline Overview

```
PDFs (Textbooks + Past Questions)
    ↓
PDF Loader (SimpleDirectoryReader + Chapter Mapping)
    ↓
Chunker (SentenceSplitter: Adaptive by Topic, Chapter-Bounded)
    ↓
Embedder (Vertex AI text-embedding-004: 768 dims)
    ↓
Firestore Vector Store (fe_vector_index collection)
```

## Firestore Document Schema

Each embedded chunk is stored as:

```json
{
  "text": "The chunk text content...",
  "embedding": [0.023, -0.045, ..., 0.156],
  "metadata": {
    "source": "textbook_1.pdf",
    "doc_type": "concept",
    "page": 42,
    "chunk_index": 5,
    "chapter": "Chapter 1: Algorithms",
    "topic": "algorithms"
  }
}
```

- `text`: Chunk content
- `embedding`: 768-dimensional vector from Vertex AI text-embedding-004
- `source`: PDF filename
- `doc_type`: "concept" or "question"
- `page`: Page number in PDF
- `chunk_index`: Sequential chunk number within source
- `chapter`: Chapter name (if chapter mapping provided)
- `topic`: Topic/subject area (e.g., "algorithms", "networks")

## Configuration

### Environment Variables (.env)

- `GOOGLE_CLOUD_PROJECT`: Firebase project ID
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON
- `FIRESTORE_COLLECTION`: Firestore collection name (default: `fe_vector_index`)
- `VERTEX_AI_LOCATION`: Vertex AI location (default: `us-central1`)

## Troubleshooting

### "Firebase credentials not found"
- Ensure `firebase-credentials.json` path in `.env` is correct
- Download from Firebase Console → Project Settings → Service Accounts

### "GOOGLE_CLOUD_PROJECT not set"
- Check `.env` file has `GOOGLE_CLOUD_PROJECT=your-project-id`
- Project ID should match your Firebase project

### Large PDF processing
- Chunking with overlap takes time; monitor logs
- Batch size is 20 nodes for embedding (adjust in `ingestion/indexer.py` if needed)

## Next Steps

- RAG retrieval pipeline (coming)
- Gemini-based question generation (coming)
- Web API for question generation (coming)

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Vertex AI Text Embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
- [Firebase Firestore](https://firebase.google.com/docs/firestore)
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Detailed design documentation
