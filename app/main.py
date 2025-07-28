
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from typing import List

from app.models.schemas import QueryRequest, QueryResponse
from app.services.document_processor import DocumentProcessor  
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.config import settings
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from typing import List

from app.models.schemas import QueryRequest, QueryResponse
from app.services.document_processor import DocumentProcessor  
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.config import settings

# Test configuration loading (you can remove these after testing)
print("🔧 Configuration loaded successfully:")
print(f"OPENROUTER_MODEL: {settings.OPENROUTER_MODEL}")  
print(f"CHUNK_SIZE: {settings.CHUNK_SIZE}")
print(f"USE_FAISS: {settings.USE_FAISS}")
print(f"REQUEST_TIMEOUT: {settings.REQUEST_TIMEOUT}")
print("-" * 50)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="HackRx Intelligent Query-Retrieval System",
    description="LLM-powered document Q&A system for insurance, legal, HR, and compliance",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global service instances
document_processor = DocumentProcessor(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
embedding_service = EmbeddingService()
vector_store = VectorStore()
llm_service = LLMService()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/")
async def root():
    return {"message": "HackRx Intelligent Query-Retrieval System", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)

async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Main endpoint for processing document queries"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        # Step 1: Process document
        chunks = await document_processor.process_document_from_url(request.documents)
        
        # Step 2: Generate embeddings for document chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = await embedding_service.get_embeddings(chunk_texts)
        
        # Step 3: Store in vector database
        await vector_store.store_documents(chunks, chunk_embeddings)
        
        # Step 4: Process each question
        question_embeddings = await embedding_service.get_embeddings(request.questions)
        
        contexts = []
        for question_embedding in question_embeddings:
            # Retrieve relevant contexts
            relevant_chunks = await vector_store.search(
                question_embedding, 
                top_k=settings.MAX_CHUNKS_FOR_CONTEXT
            )
            contexts.append(relevant_chunks)
        
        # Step 5: Generate answers using LLM
        answers = await llm_service.generate_answers(request.questions, contexts)
        
        processing_time = time.time() - start_time
        logger.info(f"Completed processing in {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error processing queries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing queries: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
