from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to document (PDF/DOCX)")
    questions: List[str] = Field(..., description="List of questions to answer")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class ProcessingResult(BaseModel):
    success: bool
    message: str
    chunks_processed: int = 0
    processing_time: float = 0.0
