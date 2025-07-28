import os
import time
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store_service import VectorStoreService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global service instances
document_service = None
embedding_service = None
llm_service = None
vector_store_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global document_service, embedding_service, llm_service, vector_store_service
    
    logger.info("🚀 Starting HackRx Intelligent Query-Retrieval System...")
    
    # Initialize services
    try:
        document_service = DocumentService()
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        vector_store_service = VectorStoreService()
        
        logger.info("✅ All services initialized successfully")
        logger.info(f"🔑 OpenRouter API Key loaded: {'Yes' if settings.OPENROUTER_API_KEY else 'No'}")
        logger.info(f"🤖 Using model: {settings.OPENROUTER_MODEL}")
        logger.info(f"⚙️ Configuration: FAISS={settings.USE_FAISS}, Chunks={settings.MAX_CHUNKS_FOR_CONTEXT}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down HackRx system...")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent document query-retrieval system for insurance policies using FastAPI + Google Gemini + FAISS vector search",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication function
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token for HackRx API access"""
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify system status"""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "openrouter_configured": bool(settings.OPENROUTER_API_KEY),
        "model": settings.OPENROUTER_MODEL
    }

# Main HackRx endpoint with proper API versioning
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_hackrx_query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Process document queries for HackRx competition
    
    - **documents**: URL to the PDF document to analyze
    - **questions**: List of questions to answer about the document
    """
    start_time = time.time()
    
    try:
        logger.info(f"📄 Processing document: {request.documents}")
        logger.info(f"❓ Questions ({len(request.questions)}): {request.questions}")
        
        # Step 1: Download and process document
        logger.info("📥 Downloading and processing document...")
        chunks = await document_service.process_document_from_url(request.documents)
        logger.info(f"📊 Document processed into {len(chunks)} chunks")
        
        # Step 2: Generate embeddings for chunks
        logger.info("🔍 Generating embeddings for document chunks...")
        chunk_embeddings = []
        for chunk in chunks:
            embedding = await embedding_service.generate_embedding(chunk["content"])
            chunk_embeddings.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": embedding
            })
        
        # Step 3: Store in vector database
        logger.info("💾 Storing embeddings in vector database...")
        await vector_store_service.store_embeddings(chunk_embeddings)
        
        # Step 4: Process each question
        answers = []
        for i, question in enumerate(request.questions, 1):
            logger.info(f"🔍 Processing question {i}/{len(request.questions)}: {question}")
            
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(question)
            
            # Search for relevant chunks
            relevant_chunks = await vector_store_service.search_similar(
                query_embedding, 
                k=settings.MAX_CHUNKS_FOR_CONTEXT
            )
            
            # Generate answer using LLM
            answer = await llm_service.generate_answer(question, relevant_chunks)
            answers.append(answer)
            
            logger.info(f"✅ Question {i} processed successfully")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"⏱️ Total processing time: {processing_time:.2f} seconds")
        
        if processing_time > settings.REQUEST_TIMEOUT:
            logger.warning(f"⚠️ Processing time ({processing_time:.2f}s) exceeded timeout ({settings.REQUEST_TIMEOUT}s)")
        
        logger.info("🎉 Request completed successfully")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error processing request after {processing_time:.2f}s: {str(e)}")
        
        # Return error in the expected format for competition
        error_answer = f"API error occurred while processing this question: {str(e)}"
        return QueryResponse(answers=[error_answer] * len(request.questions))

# Alternative endpoint without API versioning (for backward compatibility)
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_hackrx_query_legacy(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Legacy endpoint - redirects to versioned API"""
    return await process_hackrx_query(request, credentials)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "HackRx Intelligent Query-Retrieval System",
        "version": settings.APP_VERSION,
        "status": "active",
        "endpoints": {
            "health": "/health",
            "main_api": "/api/v1/hackrx/run",
            "legacy_api": "/hackrx/run"
        }
    }

# For Railway deployment - handle port dynamically
if __name__ == "__main__":
    # Get port from environment variable with fallback
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
