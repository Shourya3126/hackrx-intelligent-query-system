from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        # Use local embeddings only (since we're using OpenRouter for LLM)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info(f"Using local SentenceTransformer embeddings: {settings.EMBEDDING_MODEL_NAME}")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
