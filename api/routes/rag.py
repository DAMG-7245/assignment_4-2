from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form, Depends
from typing import List, Optional
from pydantic import BaseModel
import json
from ..services.parsing import parse_pdf
from ..services.chunking import chunk_text
from ..services.embedding import compute_embeddings
from ..services.retrieval import retrieve_context
from ..services.llm import generate_response

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    parser_type: str = "docling"  # docling, mistral_ocr, basic
    rag_method: str = "manual"    # manual, pinecone, chromadb
    chunking_strategy: str = "fixed_size"  # fixed_size, paragraph, semantic
    quarters: List[str] = []  # Specific quarters to query, e.g. ["2023-Q1", "2023-Q2"]

class QueryResponse(BaseModel):
    answer: str
    context_chunks: List[str]
    metadata: dict

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        # Retrieve context based on the query and parameters
        context_chunks, metadata = retrieve_context(
            query=request.query,
            rag_method=request.rag_method,
            chunking_strategy=request.chunking_strategy,
            quarters=request.quarters
        )
        
        # Generate response using an LLM
        answer = generate_response(request.query, context_chunks)
        
        return {
            "answer": answer,
            "context_chunks": context_chunks,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    parser_type: str = Form("docling"),
    chunking_strategy: str = Form("fixed_size"),
    quarter: str = Form(None)
):
    try:
        # Parse the uploaded PDF
        content = await file.read()
        extracted_text = parse_pdf(content, parser_type)
        
        # Chunk the extracted text
        chunks = chunk_text(extracted_text, chunking_strategy)
        
        # Compute embeddings and store in the appropriate database
        metadata = {
            "filename": file.filename,
            "quarter": quarter,
            "parser_type": parser_type,
            "chunking_strategy": chunking_strategy
        }
        
        # Store embeddings
        result = compute_embeddings(chunks, metadata)
        
        return {
            "message": "PDF processed successfully",
            "chunks_count": len(chunks),
            "metadata": metadata,
            "storage_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))