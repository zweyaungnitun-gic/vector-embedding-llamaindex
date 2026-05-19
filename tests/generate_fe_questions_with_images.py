"""Generate 2 AM and 2 PM FE questions using textbook + past question embeddings.

This script:
- Fetches contexts from `fe_textbook_vector` and `past_questions_index`.
- Filters past questions by `session` metadata (AM/PM).
- Generates 2 AM and 2 PM multiple-choice FE questions using Gemini.
- Renders each question to a PNG and combines them into a single PDF.

Run:
    python -m tests.generate_fe_questions_with_images
"""

import logging
import os
import json
import re
import urllib.request
import base64
import textwrap
from typing import List
from config.settings import get_settings
from tests.generate_fe_question_test import (
    fetch_embeddings_from_collection,
    top_k_similar,
    build_prompt,
    generate_with_gemini,
)
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join("tests", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_json_from_result(result_text: str) -> dict:
    """Strip markdown formatting and parse JSON."""
    if not result_text:
        return {}
    # Find anything that looks like JSON objects
    match = re.search(r'(\{.*\})', result_text, re.DOTALL)
    clean_text = match.group(1) if match else result_text
    clean_text = clean_text.strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}\nOffending text: {result_text[:100]}...")
        return {}


def download_mermaid_image(mermaid_syntax: str, out_path: str) -> bool:
    """Render Mermaid.js syntax to PNG using mermaid.ink API."""
    try:
        # Mermaid ink base64 payload
        payload = base64.urlsafe_b64encode(mermaid_syntax.encode('utf-8')).decode('utf-8')
        url = f"https://mermaid.ink/img/{payload}"
        urllib.request.urlretrieve(url, out_path)
        return True
    except Exception as e:
        logger.warning(f"Could not render Mermaid diagram via API: {e}")
        return False


def render_question_to_image(question_data: dict, out_path: str, width: int = 1200, padding: int = 40):
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    # Format text
    raw_text = question_data.get("question", "")
    options = question_data.get("options", [])
    for i, opt in enumerate(options):
        raw_text += f"\n  {chr(97+i)}) {opt}"
    
    # Wrap text for image rendering
    lines = []
    for paragraph in raw_text.split('\n'):
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=100))

    # getsize is deprecated in some Pillow versions; use getbbox for robust height
    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent + 6
    except Exception:
        line_height = 24

    height = padding * 2 + line_height * max(6, len(lines))

    img = Image.new("RGB", (width, max(400, height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=(0, 0, 0))
        y += line_height

    # If Mermaid image exists, append it to the bottom
    mermaid_img_path = out_path.replace(".png", "_mermaid.png")
    if question_data.get("requires_image") and question_data.get("mermaid_diagram"):
        logger.info("Attempting to render Mermaid diagram...")
        if download_mermaid_image(question_data.get("mermaid_diagram"), mermaid_img_path):
            try:
                m_img = Image.open(mermaid_img_path)
                m_width, m_height = m_img.size
                
                # Create a larger combined image
                combined = Image.new("RGB", (width, img.height + m_height + padding), color=(255, 255, 255))
                combined.paste(img, (0, 0))
                # Center the rendered diagram
                x_offset = max(padding, (width - m_width) // 2)
                combined.paste(m_img, (x_offset, img.height))
                img = combined
            except Exception as e:
                logger.error(f"Failed to append Mermaid image: {e}")

    img.save(out_path)
    return out_path


def filter_by_session(past_embs: List[dict], session: str) -> List[dict]:
    session_lower = session.lower()
    filtered = []
    for item in past_embs:
        doc = item.get("doc", {}) or {}
        # session may live under metadata when documents were stored as {'metadata': {...}}
        sess = doc.get("session") or doc.get("exam_session") or doc.get("session_name") or doc.get("metadata", {}).get("session")
        if not sess:
            # try nested common keys
            sess = doc.get("metadata", {}).get("exam_session") or doc.get("metadata", {}).get("session_name")
        if not sess:
            continue
        if isinstance(sess, str) and sess.strip().lower() == session_lower:
            filtered.append(item)
    return filtered


def main():
    settings = get_settings()
    db = settings.get_firestore_client()

    topic = "Logic gates and Boolean algebra (Please design a problem that REQUIRES a logic gate circuit diagram)"
    textbook_collection = "fe_textbook_vector"
    past_collection = "past_questions_index"

    logger.info("Embedding the topic query...")
    query_emb = settings.embed_texts(topic)[0]

    logger.info("Fetching textbook embeddings from Firestore...")
    textbook_embs = fetch_embeddings_from_collection(db, textbook_collection)
    logger.info(f"Found {len(textbook_embs)} textbook vectors")

    logger.info("Fetching past question embeddings from Firestore...")
    past_embs = fetch_embeddings_from_collection(db, past_collection)
    logger.info(f"Found {len(past_embs)} past question vectors")

    # Prepare top textbook contexts once
    top_textbook = top_k_similar(textbook_embs, query_emb, k=4)

    output_images = []

    # Just generate 1 question for testing the logic gate image
    for session_name, count in [("AM", 1)]:
        logger.info(f"Preparing {count} questions for session {session_name}...")
        session_pool = filter_by_session(past_embs, session_name)
        if not session_pool:
            logger.warning(f"No past examples found for session {session_name}. Falling back to unfiltered past questions.")
            session_pool = past_embs

        # For each question required in this session
        for idx in range(1, count + 1):
            # get top past examples for this session relative to topic
            top_past_examples = top_k_similar(session_pool, query_emb, k=3)

            prompt = build_prompt(topic, top_textbook, top_past_examples)
            logger.info(f"Generating {session_name} question {idx}...")
            try:
                result = generate_with_gemini(settings, prompt)
            except Exception as e:
                logger.error(f"Generation failed for {session_name} #{idx}: {e}")
                continue

            # Save generated text
            txt_name = f"question_{session_name}_{idx}.txt"
            txt_path = os.path.join(OUTPUT_DIR, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result)
                
            question_data = extract_json_from_result(result)
            if not question_data:
                logger.warning(f"Could not parse JSON for {session_name} question {idx}. Skipping image render.")
                continue

            # Render image
            img_name = f"question_{session_name}_{idx}.png"
            img_path = os.path.join(OUTPUT_DIR, img_name)
            render_question_to_image(question_data, img_path)
            output_images.append(img_path)
            logger.info(f"Saved question text to {txt_path} and image to {img_path}")

            # Optionally embed the generated image
            try:
                # We skip embedding the image for now if gemini-embedding-001 gives HTTP 400
                logger.info(f"Skipping Gemini multimodal embed to avoid HTTP 400.")
                pass
            except Exception as e:
                logger.debug(f"Image embedding skipped/failed: {e}")

    # Combine generated images into a single PDF in the AM..PM order collected
    try:
        # Re-filter valid images
        valid_images = [img for img in output_images if os.path.exists(img)]
        if valid_images:
            pil_images = []
            for p in valid_images:
                # Open, ensure RGB, and keep a copy in memory before passing to save
                im = Image.open(p).convert("RGB")
                pil_images.append(im)

            pdf_path = os.path.join(OUTPUT_DIR, "fe_questions_AM_PM.pdf")
            pil_images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=pil_images[1:])
            logger.info(f"Saved combined PDF to {pdf_path}")
        else:
            logger.warning("No images were generated; PDF not created.")
    except Exception as e:
        logger.error(f"Failed to create PDF from images: {e}")


if __name__ == "__main__":
    main()
