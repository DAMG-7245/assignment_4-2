from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
from typing import List, Optional
from pydantic import BaseModel

# Import service modules
from api.services.parsing import parse_document
from api.services.chunking import chunk_document
from api.services.embedding import compute_embeddings
from api.services.retrieval import retrieve_documents
from api.services.llm import generate_response

app = FastAPI(title="RAG Pipeline API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define models
class QueryRequest(BaseModel):
    query: str
    parser: str = "docling"  # Default parser
    rag_method: str = "chromadb"  # Default RAG method
    chunking_strategy: str = "semantic"  # Default chunking strategy
    quarters: List[str] = []  # Filter by specific quarters

class QueryResponse(BaseModel):
    query: str
    answer: str
    context: List[dict]
    metadata: dict

class UploadResponse(BaseModel):
    filename: str
    parser: str
    status: str
    message: str

@app.get("/")
async def root():
    return {"message": "NVIDIA RAG Pipeline API"}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    parser: str = Form("docling")
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save the uploaded file
    upload_dir = os.path.join("/app/data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Parse the document using the specified parser
    try:
        parsed_data = parse_document(file_path, parser)
        return {
            "filename": file.filename,
            "parser": parser,
            "status": "success",
            "message": f"File uploaded and parsed successfully using {parser}."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # 1. Retrieve relevant document chunks
        context_chunks = retrieve_documents(
            query=request.query,
            rag_method=request.rag_method,
            parser=request.parser,
            chunking_strategy=request.chunking_strategy,
            quarters=request.quarters
        )
        
        # 2. Generate response using LLM
        answer = generate_response(request.query, context_chunks)
        
        # 3. Return response with context
        return {
            "query": request.query,
            "answer": answer,
            "context": [
                {
                    "text": chunk["text"],
                    "doc_id": chunk["metadata"]["doc_id"],
                    "year_quarter": chunk["metadata"]["year_quarter"]
                }
                for chunk in context_chunks
            ],
            "metadata": {
                "parser": request.parser,
                "rag_method": request.rag_method,
                "chunking_strategy": request.chunking_strategy,
                "quarters": request.quarters
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/available-quarters")
async def get_available_quarters():
    try:
        # Read metadata from index files
        index_dir = "/app/data/indexes"
        available_quarters = set()
        
        for file in os.listdir(index_dir):
            if file.endswith('.json'):
                with open(os.path.join(index_dir, file), 'r') as f:
                    index_info = json.load(f)
                    
                # Extract available quarters from index metadata
                # This is a placeholder - actual implementation would depend on your data structure
                if index_info.get('quarters'):
                    available_quarters.update(index_info['quarters'])
        
        return {"quarters": sorted(list(available_quarters))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving quarters: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)