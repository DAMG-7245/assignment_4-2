# api/routes/main_routes.py

from fastapi import APIRouter, Query
from services.s3_service import list_pdfs_in_s3, get_pdf_content_from_s3
from services.parsing import parse_pdf
from services.chunking import chunk_text
from services.rag import rag_retrieval
from services.rag import rag_query_with_gemini
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List


router = APIRouter()

@router.get("/list-pdfs")
def list_pdfs():
    """
    返回 S3 里可用的 PDF key 列表
    """
    pdf_keys = list_pdfs_in_s3()
    return {"pdf_keys": pdf_keys}

@router.get("/parse-and-chunk")
def parse_and_chunk(
    pdf_key: str,
    parser_type: str = "docling",
    chunk_strategy: str = "fixed"
):
    """
    给定一个 S3 pdf_key, 用指定 parser_type 解析 -> chunk_strategy 切分
    返回分块结果(仅演示)
    """
    pdf_bytes = get_pdf_content_from_s3(pdf_key)
    parsed_text = parse_pdf(pdf_bytes, parser_type)
    chunks = chunk_text(parsed_text, chunk_strategy)
    return {"chunks": chunks, "chunks_count": len(chunks)}

@router.post("/rag-query")
def rag_query(
    user_query: str,
    chunks: list[str],
    rag_type: str = "manual",
    top_k: int = 3
):
    """
    给定用户查询 & chunks, 通过指定 rag_type 做检索, 返回最相关的top_k片段.
    这里chunks通过POST body发送(或别的方式).
    """
    top_chunks = rag_retrieval(chunks, user_query, rag_type, top_k)
    return {"top_chunks": top_chunks}



class RagGeminiRequest(BaseModel):
    user_query: str
    chunks: List[str]
    rag_type: str = "manual"
    top_k: int = 3

@router.post("/rag-ask-gemini")
def rag_ask_gemini_endpoint(payload: RagGeminiRequest):
    """
    直接调用 RAG 检索 + Gemini 生成回答.
    """
    answer = rag_query_with_gemini(
        chunks=payload.chunks,
        user_query=payload.user_query,
        rag_type=payload.rag_type,
        top_k=payload.top_k
    )
    return {"answer": answer}
