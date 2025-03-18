import os
from typing import Dict, Any, List

# Environment variables with default values
class Config:
    # Data storage
    DATA_DIR = os.getenv("DATA_DIR", "/data")
    S3_BUCKET = os.getenv("S3_BUCKET", "nvidia-reports-rag")
    
    # API keys
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    
    # Database connections
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "nvidia-reports")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "/data/chroma")
    
    # PDF parsing configurations
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Model configurations
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
    LLM_MODEL = os.getenv("LLM_MODEL", "mistral-large-latest")
    
    # Service endpoints
    API_ENDPOINT = os.getenv("API_ENDPOINT", "http://api:8000")
    
    # Available quarters (for validation)
    AVAILABLE_QUARTERS = [
        "2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4",
        "2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
        "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
        "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
        "2024-Q1", "2024-Q2"
    ]
    
    # Available PDF parsers
    PDF_PARSERS = ["basic", "docling", "mistral_ocr"]
    
    # Available RAG methods
    RAG_METHODS = ["manual", "pinecone", "chromadb"]
    
    # Available chunking strategies
    CHUNKING_STRATEGIES = ["fixed_size", "paragraph", "semantic"]
    
    @staticmethod
    def validate_quarter(quarter: str) -> bool:
        """Validate if the quarter is in the available list"""
        return quarter in Config.AVAILABLE_QUARTERS
    
    @staticmethod
    def validate_parser(parser: str) -> bool:
        """Validate if the parser is in the available list"""
        return parser in Config.PDF_PARSERS
    
    @staticmethod
    def validate_rag_method(method: str) -> bool:
        """Validate if the RAG method is in the available list"""
        return method in Config.RAG_METHODS
    
    @staticmethod
    def validate_chunking_strategy(strategy: str) -> bool:
        """Validate if the chunking strategy is in the available list"""
        return strategy in Config.CHUNKING_STRATEGIES