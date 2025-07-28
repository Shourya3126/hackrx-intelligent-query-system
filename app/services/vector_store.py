import pinecone
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        if settings.USE_FAISS or not settings.PINECONE_API_KEY:
            self._init_faiss()
        else:
            self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize Pinecone vector store"""
        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
        
        if settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                settings.PINECONE_INDEX_NAME,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
        
        self.index = pinecone.Index(settings.PINECONE_INDEX_NAME)
        self.use_pinecone = True
        logger.info("Initialized Pinecone vector store")
    
    def _init_faiss(self):
        """Initialize FAISS vector store"""
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.use_pinecone = False
        logger.info("Initialized FAISS vector store")
    
    async def store_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Store document chunks with embeddings"""
        if self.use_pinecone:
            await self._store_pinecone(chunks, embeddings)
        else:
            await self._store_faiss(chunks, embeddings)
    
    async def _store_pinecone(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Store in Pinecone"""
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"chunk_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
            })
        
        self.index.upsert(vectors)
    
    async def _store_faiss(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Store in FAISS"""
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        self.documents.extend(chunks)
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.use_pinecone:
            return await self._search_pinecone(query_embedding, top_k)
        else:
            return await self._search_faiss(query_embedding, top_k)
    
    async def _search_pinecone(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search in Pinecone"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "text": match.metadata["text"],
                "score": match.score,
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            }
            for match in results.matches
        ]
    
    async def _search_faiss(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search in FAISS"""
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx]["text"],
                    "score": float(score),
                    "metadata": self.documents[idx]["metadata"]
                })
        
        return results
