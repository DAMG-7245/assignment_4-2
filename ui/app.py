import streamlit as st
from components import (
    s3_pdf_selector,
    parser_selector,
    chunking_selector,
    parse_and_chunk_button,
    rag_query_and_answer
)

# Backend Base URL, adjust according to your deployment environment
BASE_URL = "https://python-backend-service-1077531201115.us-east1.run.app/api"

def main():
    st.title("RAG Pipeline Frontend Demo")

    # 1) Select year-quarter (from PDF mappings fetched from S3)
    st.header("1. Select Year-Quarter")
    selected_quarter = s3_pdf_selector(BASE_URL)
    st.write(f"Currently selected: {selected_quarter if selected_quarter else 'None'}")

    # 2) Select parser
    st.header("2. Select Parser")
    parser_type = parser_selector()

    # 3) Select Chunking Strategy
    st.header("3. Select Chunking Strategy")
    chunk_strategy = chunking_selector()

    # 4) Execute parsing and chunking
    st.header("4. Parse and Chunk PDF")
    parse_and_chunk_button(BASE_URL, selected_quarter, parser_type, chunk_strategy)

    # 5) Query & RAG Retrieval & (Optional) Gemini Answer
    st.header("5. Query & RAG Retrieval & (Optional) Gemini Answer")
    rag_query_and_answer(BASE_URL)

if __name__ == "__main__":
    # Initialize session state if not already present
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "chunks_count" not in st.session_state:
        st.session_state["chunks_count"] = 0
    main()
