# ITPEC FE RAG Vector Indexing Architecture

## System Overview

This document describes the architecture for building a Retrieval-Augmented Generation (RAG) system to generate ITPEC Fundamental Engineer exam questions using vector embeddings stored in Firebase Firestore.

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PDF Files                 Chunking              Embedding      │
│  (Textbooks +     ─────→  (Content-aware)  ─→  (text-embedding │
│   Past Questions)          splitting           -004)           │
│                                                  │              │
│                                                  ↓              │
│                                          Firebase Firestore     │
│                                          (Vector Index)         │
│                                                  │              │
└─────────────────────────────────────────────────┼──────────────┘
                                                  │
┌─────────────────────────────────────────────────┼──────────────┐
│                     QUERY & GENERATION PIPELINE │              │
├─────────────────────────────────────────────────┼──────────────┤
│                                                  │              │
│  User Input Topic    Embedding         Similarity Search       │
│  (e.g., "Networks")  ──────────────→  (Cosine distance,  │
│                                        k=5 nearest)       │
│                                              │            │
│                                              ↓            │
│                                      Retrieved Chunks    │
│                                      (Context)          │
│                                              │            │
│                  Context ────────────────────┤           │
│                                              ↓           │
│                                    Google Gemini LLM    │
│                              (FE MCQ Generation)        │
│                                              │           │
│                                              ↓           │
│                                        Generated Question │
│                                    (JSON: Q, Options,    │
│                                     Correct, Explanation)│
│                                                         │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
vector-embedding-llamaindex/
│
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore rules
├── ARCHITECTURE.md                     # This file
├── README.md                           # Project overview
│
├── config/
│   └── settings.py                     # Firebase initialization, env loading
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_loader.py                   # Load PDFs using SimpleDirectoryReader
│   ├── chunker.py                      # Content-aware text splitting
│   └── indexer.py                      # Embed chunks and upsert to Firestore
│
├── rag/
│   ├── __init__.py
│   ├── retriever.py                    # Firestore vector similarity search
│   └── query_engine.py                 # LlamaIndex RetrieverQueryEngine
│
├── generation/
│   ├── __init__.py
│   ├── prompts.py                      # FE MCQ generation prompts
│   └── generator.py                    # Gemini-based question generator
│
├── data/
│   ├── textbooks/                      # FE textbooks (PDF) — not committed
│   │   ├── textbook_1.pdf
│   │   └── textbook_2.pdf
│   │
│   └── past_questions/                 # Past exam papers (PDF) — not committed
│       └── past_exams.pdf
│
└── main.py                             # CLI entry point
```

---

## Data Pipeline

### 1. PDF Ingestion & Loading

**Module:** `ingestion/pdf_loader.py`

- Uses LlamaIndex `SimpleDirectoryReader` with `PDFReader`
- Reads PDFs from `data/textbooks/` and `data/past_questions/`
- Extracts text and retains document metadata (filename, page number)

**Output:**
- List of `Document` objects with:
  - `text`: Raw text content
  - `metadata`: `{"source": "textbook_1", "page": 5, "doc_type": "concept"}`

---

### 2. Intelligent Chunking

**Module:** `ingestion/chunker.py`

Different strategies for textbooks vs. past questions:

#### Strategy A: Textbooks (Concept Content)
- **Splitter:** LlamaIndex `SentenceSplitter`
- **Chunk Size:** 512 tokens
- **Overlap:** 64 tokens (preserve context)
- **Rationale:** Concepts are continuous prose; sentences preserve semantic boundaries

#### Strategy B: Past Questions (Q&A Pairs)
- **Splitter:** Custom logic (regex-based or heuristic)
- **Atomic Units:** Question + 4 options + correct answer + explanation = 1 chunk
- **Rationale:** Keep exam questions intact; querying single Q&A pairs is more useful than fragmentary question text

**Example Q&A Chunk:**
```
Question: What is the time complexity of binary search?
A) O(n)
B) O(log n)
C) O(n log n)
D) O(n²)
Correct: B
Explanation: Binary search divides the search space in half each iteration.
```

**Output:**
- List of `TextNode` objects with chunk text and inherited metadata

---

### 3. Embedding Generation

**Module:** `ingestion/indexer.py`

- **Embedding Model:** Google `text-embedding-004` (768 dimensions)
- **Provider:** LlamaIndex `GoogleGenerativeAIEmbedding`
- **Batch Processing:** Chunks embedded in batches of 50 to optimize API calls

**Output:**
- Each chunk gets a 768-dim embedding vector
- Ready for Firestore upsert

---

### 4. Vector Storage (Firestore)

**Module:** `config/settings.py` (initialization)

**Collection Name:** `fe_vector_index`

**Document Schema:**

```json
{
  "id": "auto-generated",
  "text": "Question: What is...",
  "embedding": [0.023, -0.045, ..., 0.156],  // 768 floats
  "metadata": {
    "source": "textbook_1|textbook_2|past_questions",
    "doc_type": "concept|question",
    "page": 42,
    "chunk_index": 5,
    "topic": "algorithms|networks|databases|security|...",
    "timestamp": "2026-05-15T10:30:00Z"
  }
}
```

**Indexing in Firestore:**
- Firestore does NOT natively support vector search
- Use LlamaIndex `FirestoreVectorStore` which implements client-side similarity search
- Alternatively: Use Google Cloud Vertex AI Matching Engine for server-side vector search (advanced)

---

## RAG (Retrieval-Augmented Generation) Pipeline

### 1. Query Embedding

**Module:** `rag/retriever.py`

- User provides a topic: `"Networks and routing protocols"`
- Embed the query using the same `text-embedding-004` model
- Result: 768-dim query vector

---

### 2. Similarity Search in Firestore

**Module:** `rag/retriever.py`

Using LlamaIndex `FirestoreVectorStore`:

```python
retriever = FirestoreVectorStore(
    collection="fe_vector_index",
    embedding_field="embedding",
    text_field="text"
)

nodes = retriever.retrieve(
    query_embedding=query_vector,
    similarity_top_k=5
)
```

- Computes cosine similarity between query vector and all stored embeddings
- Returns top-5 most relevant chunks
- **Note:** Client-side search; scale up with Vertex AI if needed (>10k documents)

**Output:**
- 5 `Node` objects with highest similarity scores
- Ready to use as RAG context

---

### 3. Context Aggregation

**Module:** `rag/query_engine.py`

Combine retrieved chunks into a single context string:

```
CONTEXT:
---
Chunk 1: [concept from textbook]
Chunk 2: [related past exam question]
Chunk 3: [another concept]
---
```

---

### 4. Question Generation via Gemini

**Module:** `generation/generator.py`

Send to Google Gemini with a structured prompt:

**Prompt Template:**
```
You are an ITPEC FE exam question generator.

CONTEXT:
{retrieved_context}

TASK:
Generate 1 multiple-choice question in ITPEC FE exam format based on the context above.

OUTPUT FORMAT (JSON):
{
  "question": "Question text here?",
  "options": {
    "A": "Option A text",
    "B": "Option B text",
    "C": "Option C text",
    "D": "Option D text"
  },
  "correct": "B",
  "explanation": "Explanation why B is correct."
}

Generate the question now:
```

**Model:** `gemini-1.5-pro` or `gemini-2.0-flash-exp`
**Temperature:** 0.7 (balanced creativity and consistency)
**Response Format:** JSON (parsed using `pydantic`)

---

### 5. Structured Output

**Module:** `generation/generator.py` / `generation/prompts.py`

**Data Model (Pydantic):**

```python
from pydantic import BaseModel
from typing import Dict

class FEQuestion(BaseModel):
    question: str
    options: Dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct: str             # "A", "B", "C", or "D"
    explanation: str
```

**Example Output:**

```json
{
  "question": "In a TCP connection, what is the purpose of the three-way handshake?",
  "options": {
    "A": "To authenticate the client and server",
    "B": "To establish a reliable, ordered connection and synchronize sequence numbers",
    "C": "To compress data before transmission",
    "D": "To encrypt the entire communication"
  },
  "correct": "B",
  "explanation": "The three-way handshake (SYN, SYN-ACK, ACK) initializes the TCP state machines, synchronizes sequence numbers, and confirms both sides are ready to communicate. This ensures reliable, in-order delivery."
}
```

---

## Embedding Methodology

### Choice: Google Text-Embedding-004

**Why:**
- **Dimension:** 768 (good balance of expressiveness and query speed)
- **Quality:** High semantic understanding, trained on diverse text corpus
- **Integration:** Native LlamaIndex support via `GoogleGenerativeAIEmbedding`
- **Cost:** Free tier available for prototyping

**Alternatives Considered:**
- **OpenAI `text-embedding-3-small`:** Good but requires OpenAI API key
- **Sentence Transformers:** Free and lightweight, but lower quality

### Similarity Metric

- **Distance:** Cosine similarity (standard for embeddings)
- **Top-K:** Retrieve 5 most relevant chunks
- **Threshold:** No hard cutoff; use all top-5

---

## Environment Variables

**File:** `.env.example`

```bash
# Google API (Gemini & Embeddings)
GOOGLE_API_KEY=your-google-api-key-here

# Firebase Admin SDK
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_CREDENTIALS_PATH=./firebase-credentials.json
```

**Setup:**
1. Create a `.env` file (copy from `.env.example`)
2. Obtain `GOOGLE_API_KEY` from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Download Firebase service account JSON from Firebase Console → Project Settings → Service Accounts
4. Place JSON file at `firebase-credentials.json` (in repo root)

---

## Verification & Testing

### 1. Ingest Test Data

```bash
python main.py ingest --source data/textbooks/ --source data/past_questions/
```

- Check Firestore: Should have ~500+ documents in `fe_vector_index` collection
- Verify embeddings are 768-dim arrays (not null)

### 2. Test Retrieval

```bash
python main.py query --topic "networks and routing"
```

- Should return 5 chunks
- Verify chunks are relevant to "networks"

### 3. Test Question Generation

```bash
python main.py generate --topic "databases" --count 3
```

- Should generate 3 FE-style questions
- Verify JSON structure is valid
- Manually check question quality

---

## Scaling Considerations

### Current: Client-Side Vector Search
- **Limit:** ~10,000 documents (Firestore read costs become significant)
- **Latency:** 1-2 seconds per query (all documents searched)
- **Cost:** Low for small datasets

### Future: Server-Side Vector Search
- Migrate to **Google Cloud Vertex AI Matching Engine**
- Sub-100ms query latency
- Scales to millions of documents
- Requires slight code refactor (use `VertexAIVectorStore` instead of `FirestoreVectorStore`)

---

## Dependencies

See `requirements.txt`:
- `llama-index-core`: Core RAG framework
- `llama-index-vector-stores-firestore`: Firestore integration
- `llama-index-embeddings-google`: Text embedding models
- `llama-index-llms-google-genai`: Gemini LLM integration
- `firebase-admin`: Firebase Admin SDK
- `google-generativeai`: Direct Gemini API (backup)
- `pypdf`: PDF text extraction
- `pydantic`: Data validation
- `python-dotenv`: Environment variable loading

---

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Google Generative AI Docs](https://ai.google.dev/)
- [Firebase Firestore Documentation](https://firebase.google.com/docs/firestore)
- [ITPEC FE Exam Format](https://www.itpec.org/)
