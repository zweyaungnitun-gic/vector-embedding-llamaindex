import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai

load_dotenv()

class Settings:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.firestore_collection = os.getenv("FIRESTORE_COLLECTION", "fe_vector_index")
        self.custom_database = os.getenv("CUSTOM_DATABASE", "it-exam")

        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        if not self.credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env")
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Firebase credentials not found at {self.credentials_path}")

        self._init_gemini()
        self._init_firebase()

    def _init_gemini(self):
        """Initialize Google Generative AI (Gemini) with API key."""
        self.genai_client = genai.Client(api_key=self.google_api_key)

    def _init_firebase(self):
        """Initialize Firebase Admin SDK for Firestore."""
        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(self.credentials_path)
            firebase_admin.initialize_app(cred)

    def get_firestore_client(self):
        """Get Firestore client for the custom database."""
        return firestore.client(database_id=self.custom_database)

    def embed_texts(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        """Embed texts using Gemini text-embedding-004 model (v1 stable API).

        Args:
            texts: String or list of strings to embed
            task_type: Task type for embeddings (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)

        Returns:
            List[List[float]] - Embeddings (768-dim vectors)
        """
        response = self.genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts if isinstance(texts, list) else [texts]
        )

        # Extract embeddings from response
        # response.embeddings is a list of ContentEmbedding objects
        embeddings = [embedding.values for embedding in response.embeddings]

        return embeddings

    def embed_images(self, image_data_list):
        """Embed images using Gemini's multimodal embedding capability.

        Args:
            image_data_list: List of image data (file paths, base64, or bytes)

        Returns:
            List[List[float]] - Image embeddings (768-dim vectors)
        """
        import base64
        from pathlib import Path
        
        embeddings = []
        for image_data in image_data_list:
            try:
                # If it's a file path, read and encode
                if isinstance(image_data, str) and Path(image_data).exists():
                    with open(image_data, "rb") as img_file:
                        image_bytes = img_file.read()
                    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
                    mime_type = "image/jpeg" if image_data.lower().endswith((".jpg", ".jpeg")) else "image/png"
                # If it's already base64, use as is
                elif isinstance(image_data, str):
                    image_b64 = image_data
                    mime_type = "image/jpeg"
                # If it's bytes, encode
                else:
                    image_b64 = base64.standard_b64encode(image_data).decode("utf-8")
                    mime_type = "image/jpeg"
                
                # Create image content with MIME type
                image_content = {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_b64
                    }
                }
                
                # Embed using Gemini's multimodal capability
                response = self.genai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=[image_content]
                )
                
                embedding = response.embeddings[0].values
                embeddings.append(embedding)
                
            except Exception as e:
                raise ValueError(f"Error embedding image: {e}")
        
        return embeddings

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
