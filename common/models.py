from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum

class ParserType(str, Enum):
    BASIC = "basic"
    DOCLING = "docling"
    MISTRAL_OCR = "mistral_ocr"

class RAGMethod(str, Enum):
    MANUAL = "manual"
    PINECONE = "pinecone"
    CHROMADB = "chromadb"

class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"

class Quarter(str, Enum):
    Q1_2020 = "2020-Q1"
    Q2_2020 = "2020-Q2"
    Q3_2020 = "2020-Q3"
    Q4_2020 = "2020-Q4"
    Q1_2021 = "2021-Q1"
    Q2_2021 = "2021-Q2"
    Q3_2021 = "2021-Q3"
    Q4_2021 = "2021-Q4"
    Q1_2022 = "2022-Q1"
    Q2_2022 = "2022-Q2"
    Q3_2022 = "2022-Q3"
    Q4_2022 = "2022-Q4"
    Q1_2023 = "2023-Q1"
    Q2_2023 = "2023-Q2"
    Q3_2023 = "2023-Q3"
    Q4_2023 = "2023-Q4"
    Q1_2024 = "2024-Q1"
    Q2_2024 = "2024-Q2"

class Document(BaseModel):
    """A document model for storing metadata and content"""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class TextChunk(BaseModel):
    """A text chunk with its metadata"""
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('embedding')
    def validate_embedding_length(cls, v):
        """Validate that the embedding has the correct dimensionality"""
        if v is not None and len(v) != 1024:  # Example for Mistral embeddings
            raise ValueError(f"Embedding must have 1024 dimensions, got {len(v)}")
        return v

class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    parser_type: ParserType = ParserType.DOCLING
    rag_method: RAGMethod = RAGMethod.MANUAL
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    quarters: List[Quarter] = Field(default_factory=list)

class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    context_chunks: List[str]
    metadata: Dict[str, Any]

class PDFUploadResponse(BaseModel):
    """PDF upload response model"""
    message: str
    chunks_count: int
    metadata: Dict[str, Any]
    storage_result: Dict[str, Any]