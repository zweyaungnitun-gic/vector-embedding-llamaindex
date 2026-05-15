# Testing Vector Embedding Pipeline

## Quick Test: `test_embed.py`

This script tests the complete embedding and Firestore storage pipeline without needing any PDFs.

### What It Does

1. **Creates 5 sample TextNodes** with FE exam topics:
   - Algorithms (binary search)
   - Data Structures (stacks vs queues)
   - Networks (TCP three-way handshake)
   - Databases (ACID properties)
   - Security (symmetric vs asymmetric encryption)

2. **Embeds using Vertex AI** (`text-embedding-004`)
   - Each text gets a 768-dimensional vector
   - Processed in batches of 20

3. **Stores in Firestore** test collection (`fe_vector_test`)
   - Keeps test data separate from real ingestion

4. **Verifies stored data**:
   - Reads back all 5 documents
   - Confirms embeddings are 768-dim
   - Confirms metadata is preserved

### Prerequisites

```bash
# 1. Setup environment (if not already done)
cp .env.example .env
# Edit .env with your credentials:
#   GOOGLE_CLOUD_PROJECT=your-project-id
#   GOOGLE_APPLICATION_CREDENTIALS=./firebase-credentials.json

# 2. Place Firebase credentials
cp /path/to/firebase-credentials.json ./firebase-credentials.json

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Test

```bash
python test_embed.py
```

### Expected Output

```
======================================================================
Starting embedding test with sample data...
======================================================================

Step 1: Creating 5 sample TextNode objects...
✓ Created 5 sample nodes

Step 2: Embedding and storing in Firestore collection 'fe_vector_test'...
2026-05-15 14:23:45 - ingestion.indexer - INFO - Embedding batch 1 (5 nodes)...
✓ Successfully stored 5 documents in Firestore collection: fe_vector_test

Step 3: Verifying data in Firestore...

Verification Results:
======================================================================
  [1] ✓ topic=algorithms           | embedding_dims=768 | text=Binary search is an effici...
  [2] ✓ topic=data_structures      | embedding_dims=768 | text=Stacks and queues are fu...
  [3] ✓ topic=networks             | embedding_dims=768 | text=The TCP three-way hands...
  [4] ✓ topic=databases            | embedding_dims=768 | text=ACID properties ensure da...
  [5] ✓ topic=security             | embedding_dims=768 | text=Symmetric and asymmetric ...
======================================================================

======================================================================
✓ TEST PASSED: All embeddings stored and verified!
======================================================================
```

## Troubleshooting

### "GOOGLE_CLOUD_PROJECT not set"
- Check `.env` file exists and has `GOOGLE_CLOUD_PROJECT=your-project-id`

### "Firebase credentials not found"
- Check `firebase-credentials.json` path in `.env` is correct
- Download from Firebase Console → Project Settings → Service Accounts

### "PermissionError: Invalid service account"
- Verify service account has Firestore write permissions
- Check Firebase project ID matches in credentials file

### "Rate limited by Vertex AI"
- Vertex AI embeddings API has quota limits
- Try again in a few moments or check Cloud Console for quota

## Next Steps

Once the test passes:
1. Update `data/chapter_mapping.json` with your textbook structure
2. Place PDF files in `data/textbooks/`
3. Run full ingestion:
   ```bash
   python main.py ingest --path data/textbooks/ --mapping data/chapter_mapping.json
   ```

## Cleanup (Optional)

To delete test data from Firestore:
```bash
firebase firestore delete fe_vector_test --all-collections
```

Or directly from the Firebase Console: Firestore → Database → fe_vector_test collection → Delete collection
