"""Test: generate an FE-style question using textbook and past-question embeddings.

This script is a runnable test that does the following:
- Loads settings from `config.settings`
- Retrieves all documents from two Firestore collections: `fe_textbook_vector` and `past_questions_index`
- Computes cosine similarity between a user topic prompt and stored embeddings
- Selects top contexts from both collections and constructs a prompt
- Calls Google Gemini (affordable model `gemini-1.5-mini`) to generate a new FE multiple-choice question

Notes:
- Make sure your `.env` is populated and Firestore collections exist and contain `embedding` vectors.
- Install requirements: `pip install -r requirements.txt`

Run:
    python -m tests.generate_fe_question_test
"""

import logging
from typing import List, Dict, Any, Tuple
import math
import numpy as np
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fetch_embeddings_from_collection(db, collection_name: str, page_size: int = 500, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch embeddings from Firestore collection using pagination and retry logic to
    avoid server-side query timeouts on large collections.

    Args:
        db: Firestore client
        collection_name: Name of the collection to read
        page_size: Number of documents to fetch per page
        max_retries: Number of retry attempts on transient errors

    Returns:
        List of dicts with keys: id, embedding, doc
    """
    results: List[Dict[str, Any]] = []

    collection_ref = db.collection(collection_name)
    next_query = collection_ref.limit(page_size)

    while next_query is not None:
        attempt = 0
        while True:
            try:
                docs = list(next_query.stream())
                break
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    raise
                backoff = 2 ** attempt
                logger.warning(f"Transient Firestore error when reading {collection_name}: {exc}. Retrying in {backoff}s (attempt {attempt}/{max_retries})")
                import time
                time.sleep(backoff)

        if not docs:
            break

        for d in docs:
            try:
                data = d.to_dict()
            except Exception:
                continue
            if not data:
                continue
            emb = data.get("embedding")
            if not emb:
                continue
            results.append({"id": d.id, "embedding": emb, "doc": data})

        # If we retrieved fewer than page_size docs, we're done
        if len(docs) < page_size:
            break

        # Prepare next query starting after the last document
        last_doc = docs[-1]
        try:
            next_query = collection_ref.start_after(last_doc).limit(page_size)
        except Exception:
            # Fallback: stop pagination if start_after fails
            break

    return results


def top_k_similar(embeddings_source: List[Dict[str, Any]], query_emb: List[float], k: int = 3) -> List[Dict[str, Any]]:
    scored = []
    for item in embeddings_source:
        sim = cosine_similarity(query_emb, item["embedding"])
        scored.append((sim, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


def build_prompt(topic: str, textbook_contexts: List[Dict[str, Any]], past_examples: List[Dict[str, Any]]) -> str:
    prompt = []
    prompt.append(f"Generate one FE multiple-choice question (4 options, single correct) on the topic: {topic}.")
    prompt.append("Use the following textbook context for facts:")
    for i, node in enumerate(textbook_contexts, start=1):
        text = node.get("doc", {}).get("text") or node.get("doc", {}).get("content") or ""
        prompt.append(f"[Textbook-{i}] {text[:500]}")

    prompt.append("\nUse these past exam questions as style examples (do not copy them):")
    for i, ex in enumerate(past_examples, start=1):
        ex_text = ex.get("doc", {}).get("text") or ex.get("doc", {}).get("content") or ex.get("doc", {}).get("question") or "(image)"
        prompt.append(f"[Past-{i}] {ex_text[:400]}")

    prompt.append("\nProduce output in JSON with keys: question, options (list of 4 strings), answer_index (0-3), explanation.")
    prompt.append("Make distractors plausible and aligned with FE exam style.")

    return "\n\n".join(prompt)


def generate_with_gemini(settings, prompt: str) -> str:
    # Try a few API call patterns depending on genai SDK available
    client = settings.genai_client
    model = "gemini-2.5-flash"  # Try gemini-2.5-flash (or gemini-2.0-flash / gemini-pro)
    # Try several genai SDK call patterns and parse common response shapes
    errors = []

    # Pattern A: responses.create(model=..., input=...)
    try:
        if hasattr(client, "responses"):
            resp = client.responses.create(model=model, input=prompt)
            # Common shapes: resp.output_text, resp.output, resp.outputs
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text
            if hasattr(resp, "output") and resp.output:
                return str(resp.output)
            if hasattr(resp, "outputs") and resp.outputs:
                # try to extract textual parts
                try:
                    out = resp.outputs[0]
                    if hasattr(out, "content"):
                        # content can be list of dicts
                        content = out.content
                        if isinstance(content, (list, tuple)) and len(content) > 0:
                            # look for text fields
                            first = content[0]
                            text = None
                            if isinstance(first, dict):
                                text = first.get("text") or first.get("text_output")
                            if text:
                                return text
                except Exception:
                    pass
    except Exception as e:
        errors.append(f"responses.create failed: {e}")

    # Pattern B: models.generate_content(model=..., prompt=...)
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            # SDK expects `contents` to be a list of strings or media objects; pass plain strings
            try_contents = [prompt]
            resp = client.models.generate_content(model=model, contents=try_contents)
            if hasattr(resp, "candidates") and resp.candidates:
                candidate = resp.candidates[0]
                # candidate may have content or text
                # candidate.content may be a string or structured object
                if hasattr(candidate, "content") and candidate.content:
                    try:
                        return candidate.content if isinstance(candidate.content, str) else str(candidate.content)
                    except Exception:
                        return str(candidate.content)
                if hasattr(candidate, "output_text") and candidate.output_text:
                    return candidate.output_text
            if hasattr(resp, "output"):
                return str(resp.output)
    except Exception as e:
        errors.append(f"models.generate_content failed: {e}")

    # Pattern C: models.generate (different SDKs)
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate"):
            resp = client.models.generate(model=model, input=prompt)
            # parse common fields
            if hasattr(resp, "output"):
                return str(resp.output)
            if hasattr(resp, "candidates") and resp.candidates:
                return resp.candidates[0].content
    except Exception as e:
        errors.append(f"models.generate failed: {e}")

    # Pattern D: older client method 'generate' on client
    try:
        if hasattr(client, "generate"):
            resp = client.generate(model=model, prompt=prompt)
            if isinstance(resp, str):
                return resp
            if hasattr(resp, "text"):
                return resp.text
    except Exception as e:
        errors.append(f"client.generate failed: {e}")

    # If none worked, raise with accumulated errors for debugging
    raise RuntimeError("Could not call Gemini generation - tried multiple SDK patterns. Errors: " + " | ".join(errors))


def main():
    settings = get_settings()
    db = settings.get_firestore_client()

    # Collections (adjust as necessary)
    textbook_collection = "fe_textbook_vector"
    past_collection = "past_questions_index"

    topic = "Networks"  # example topic to generate question for

    logger.info("Embedding the query/topic...")
    query_emb = settings.embed_texts(topic)[0]

    logger.info("Fetching textbook embeddings from Firestore...")
    textbook_embs = fetch_embeddings_from_collection(db, textbook_collection)
    logger.info(f"Found {len(textbook_embs)} textbook vectors")

    logger.info("Fetching past question embeddings from Firestore...")
    past_embs = fetch_embeddings_from_collection(db, past_collection)
    logger.info(f"Found {len(past_embs)} past question vectors")

    top_textbook = top_k_similar(textbook_embs, query_emb, k=3)
    top_past = top_k_similar(past_embs, query_emb, k=3)

    prompt = build_prompt(topic, top_textbook, top_past)
    logger.info("Prompt built. Calling Gemini to generate question...")

    result = generate_with_gemini(settings, prompt)
    print("\n=== GENERATED QUESTION ===\n")
    print(result)


if __name__ == "__main__":
    main()
