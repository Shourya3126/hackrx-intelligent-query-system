import time
import logging
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import QueryRequest, QueryResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx Intelligent Query-Retrieval System",
    description="LLM-powered document Q&A system for insurance policies and compliance",
    version="1.0.0",
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Initialize your services as global variables (or via dependency injection if preferred)
document_processor = DocumentProcessor(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
embedding_service = EmbeddingService()
vector_store = VectorStore()
llm_service = LLMService()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token for API access."""
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {"message": "HackRx Intelligent Query-Retrieval System", "status": "active"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": "HackRx Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "openrouter_configured": bool(settings.OPENROUTER_API_KEY),
        "model": settings.OPENROUTER_MODEL,
    }


@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_queries_v1(
    request: QueryRequest,
    token: str = Depends(verify_token),
) -> QueryResponse:
    """Versioned endpoint to process queries."""
    return await _process_queries(request)


@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries_hackrx(
    request: QueryRequest,
    token: str = Depends(verify_token),
) -> QueryResponse:
    """Official HackRx competition endpoint."""
    return await _process_queries(request)


async def _process_queries(request: QueryRequest) -> QueryResponse:
    start_time = time.time()
    try:
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")

        # Step 1: Download and process the document into chunks
        chunks = await document_processor.process_document_from_url(request.documents)

        # Step 2: Generate embeddings for the document chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = await embedding_service.get_embeddings(chunk_texts)

        # Step 3: Store embeddings in vector store
        await vector_store.store_documents(chunks, chunk_embeddings)

        # Step 4: Generate embeddings for questions
        question_embeddings = await embedding_service.get_embeddings(request.questions)

        # Step 5: Retrieve relevant contexts for each question
        contexts = []
        for q_emb in question_embeddings:
            relevant_chunks = await vector_store.search(q_emb, top_k=settings.MAX_CHUNKS_FOR_CONTEXT)
            contexts.append(relevant_chunks)

        # Step 6: Generate answers using the language model
        answers = await llm_service.generate_answers(request.questions, contexts)

        elapsed = time.time() - start_time
        logger.info(f"Completed processing in {elapsed:.2f} seconds")

        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing queries: {str(e)}")


if __name__ == "__main__":
    import os
    import uvicorn

    # Use PORT env variable if set, fallback to 8000
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting app on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
