# rag-pipeline/ui/app.py

import streamlit as st
from components import (
    s3_pdf_selector,
    parser_selector,
    chunking_selector,
    parse_and_chunk_button,
    rag_query_and_answer
)

# 你后端的 Base URL (Docker Compose 或本地)
BASE_URL = "http://localhost:8000/api"

def main():
    st.title("RAG Pipeline 前端 Demo")

    # 1) 选择 S3中的 PDF
    st.header("1. 选择 PDF")
    selected_pdf = s3_pdf_selector(BASE_URL)
    st.write(f"当前选定: {selected_pdf if selected_pdf else '无'}")

    # 2) 选择解析器
    st.header("2. 选择解析器")
    parser_type = parser_selector()

    # 3) 选择 Chunking 策略
    st.header("3. 选择 Chunking 策略")
    chunk_strategy = chunking_selector()

    # 4) 执行 解析并切分
    st.header("4. 解析并切分 PDF")
    parse_and_chunk_button(BASE_URL, selected_pdf, parser_type, chunk_strategy)

    # 5) 查询 + RAG + Gemini
    st.header("5. 查询 & RAG检索 & (可选) Gemini回答")
    rag_query_and_answer(BASE_URL)

if __name__ == "__main__":
    # 初始化 session_state
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "chunks_count" not in st.session_state:
        st.session_state["chunks_count"] = 0

    main()
