import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration for HackRx
    API_KEY: str = "8c8cb782ead692a2056fd1494ecf1ee2ebd90d05d4d4dbd8e25bcfb03355bf00"
    
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = "openai/gpt-4o-mini"  # Best for document analysis
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    
    
    # Embedding Configuration (Using local for cost-effectiveness)
    USE_LOCAL_EMBEDDINGS: bool = True
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"  # Local sentence transformer
    
    # Vector Store Configuration
    USE_FAISS: bool = True
    FAISS_INDEX_PATH: str = "./faiss_index"
    
    # Alternative: Pinecone (for production scaling)
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "hackrx-docs"
    
    # Document Processing Settings
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    MAX_CHUNKS_FOR_CONTEXT: int = 5
    
    # LLM Generation Settings
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.1
    
    # Performance Settings
    REQUEST_TIMEOUT: int = 25  # seconds (under 30s requirement)
    MAX_RETRIES: int = 3
    BATCH_SIZE: int = 100  # For embedding generation
    
    # Application Settings
    APP_NAME: str = "HackRx Intelligent Query-Retrieval System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Optional: Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # Optional: Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()
