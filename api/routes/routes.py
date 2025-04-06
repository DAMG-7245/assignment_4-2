from fastapi import APIRouter, HTTPException
from services.s3_service import  get_quarterly_report_mapping_from_s3,  get_s3_presigned_url
from services.parsing import parse_pdf_any
from services.chunking import chunk_text
from services.rag import rag_retrieval, rag_query_with_gemini
from pydantic import BaseModel
from typing import List
import requests

router = APIRouter()

@router.get("/list-pdfs")
def list_pdfs():
    """
    返回 S3 里存放的 PDF 映射 {YearQuarter: PDF URL}
    """
    mapping = get_quarterly_report_mapping_from_s3()
    return {"mapping": mapping}

@router.get("/parse-and-chunk")
def parse_and_chunk(
    quarter: str,
    parser_type: str = "docling_url",  # 支持："mistral_ocr_url", "docling_url", "jina_bytes"
    chunk_strategy: str = "fixed"
):
    """
    根据用户选择的年份季度，从 S3 映射中获取对应的 PDF 信息，
    然后根据选择的解析方式解析 PDF，并按照指定策略进行切分。
    """
    mapping = get_quarterly_report_mapping_from_s3()
    if quarter not in mapping:
        raise HTTPException(status_code=404, detail=f"未找到季度 {quarter} 对应的 PDF URL")
    
    pdf_url = mapping[quarter]

    # 如果 pdf_url 已经以 "http" 开头，则认为它是完整 URL，
    # 否则生成预签名 URL
    if pdf_url.startswith("http"):
        public_pdf_url = pdf_url
    else:
        public_pdf_url = get_s3_presigned_url(pdf_url)
        if not public_pdf_url:
            raise HTTPException(status_code=400, detail="无法生成 PDF 的预签名 URL")
    
    if parser_type in ("mistral_ocr_url", "docling_url"):
        parsed_text = parse_pdf_any(pdf_url=public_pdf_url, parser_type=parser_type)
    elif parser_type == "jina_bytes":
        # 对于 jina_bytes 方式，先下载 PDF bytes
        pdf_resp = requests.get(public_pdf_url)
        if pdf_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="无法下载 PDF 文件")
        pdf_bytes = pdf_resp.content
        parsed_text = parse_pdf_any(parser_type="jina_bytes", pdf_bytes=pdf_bytes)
    else:
        raise HTTPException(status_code=400, detail=f"不支持的 parser_type: {parser_type}")

    chunks = chunk_text(parsed_text, chunk_strategy)
    return {"chunks": chunks, "chunks_count": len(chunks)}

class RagQueryRequest(BaseModel):
    user_query: str
    chunks: List[str]
    rag_type: str = "manual"
    top_k: int = 3

@router.post("/rag-query")
def rag_query_endpoint(payload: RagQueryRequest):
    top_chunks = rag_retrieval(payload.chunks, payload.user_query, payload.rag_type, payload.top_k)
    return {"top_chunks": top_chunks}

class RagGeminiRequest(BaseModel):
    user_query: str
    chunks: List[str]
    rag_type: str = "manual"
    top_k: int = 3

@router.post("/rag-ask-gemini")
def rag_ask_gemini_endpoint(payload: RagGeminiRequest):
    """
    使用 RAG 检索后，将最相关的文本块与用户问题组合，
    调用 Gemini 生成最终回答。
    """
    answer = rag_query_with_gemini(
        chunks=payload.chunks,
        user_query=payload.user_query,
        rag_type=payload.rag_type,
        top_k=payload.top_k
    )
    return {"answer": answer}
