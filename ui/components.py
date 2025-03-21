import streamlit as st
import requests

def s3_pdf_selector(base_url: str) -> str:
    """
    Component 1: Retrieve a mapping of PDFs from S3 ({YearQuarter: PDF URL})
    from a backend, and allow the user to select a year-quarter (e.g., "2024Q1").
    :param base_url: Backend API URL (e.g., http://localhost:8000/api)
    :return: The user-selected year-quarter (str)
    """
    mapping = {}
    with st.spinner("Loading PDF mappings..."):
        try:
            resp = requests.get(f"{base_url}/list-pdfs")
            resp.raise_for_status()
            mapping = resp.json().get("mapping", {})
        except Exception as e:
            st.error(f"Failed to fetch PDF mappings: {e}")

    if not mapping:
        st.warning("No PDF mappings found in S3, please check or wait for Airflow to crawl.")
        return ""

    # User selects a key (year-quarter) from the mapping
    selected_quarter = st.selectbox("Select a year-quarter:", list(mapping.keys()))
    return selected_quarter

def parser_selector() -> str:
    """
    Component 2: Allows the user to select a PDF parsing method.
    Currently supports:
      - mistral_ocr_url: Parse using Mistral OCR via URL
      - docling_url: Parse using Docling via URL
      - jina_bytes: Parse using Jina (requires downloading PDF bytes)
    """
    parser_type = st.selectbox("Select a parser", ["mistral_ocr_url", "docling_url", "jina_bytes"])
    return parser_type

def chunking_selector() -> str:
    """
    Component 3: Allows the user to select a chunking strategy (fixed, paragraph, sentence).
    """
    chunk_strategy = st.selectbox("Select a chunking strategy", ["fixed", "paragraph", "sentence"])
    return chunk_strategy

def parse_and_chunk_button(base_url: str, quarter: str, parser_type: str, chunk_strategy: str):
    """
    Component 4: When the user clicks the button, call the backend /parse-and-chunk endpoint,
    passing the selected year-quarter, parser type, and chunking strategy,
    and store the returned chunks in the session_state.
    """
    if st.button("Parse and Chunk"):
        if not quarter:
            st.warning("Please select a year-quarter first")
            return
        with st.spinner("Parsing and chunking..."):
            try:
                parse_resp = requests.get(
                    f"{base_url}/parse-and-chunk",
                    params={
                        "quarter": quarter,
                        "parser_type": parser_type,
                        "chunk_strategy": chunk_strategy,
                    }
                )
                parse_resp.raise_for_status()
                data = parse_resp.json()
                st.session_state["chunks"] = data["chunks"]
                st.session_state["chunks_count"] = data["chunks_count"]
                st.success(f"Parsing successful, obtained {data['chunks_count']} text chunks!")
            except Exception as e:
                st.error(f"Failed to parse: {e}")

def rag_query_and_answer(base_url: str):
    """
    Component 5: Allows the user to input a query, choose a RAG retrieval method,
    and opt to use Gemini to generate the final answer.
    """
    chunks = st.session_state.get("chunks", [])
    if not chunks:
        st.info("Please perform parse and chunk first to obtain text chunks.")
        return

    st.subheader("Query and Answer")
    user_query = st.text_input("Enter your question:", "")
    rag_type = st.selectbox("RAG Method", ["manual", "pinecone", "chromadb"])
    top_k = st.slider("Retrieve Top K", 1, 5, 3)
    use_gemini = st.checkbox("Use Gemini to generate the final answer")

    if st.button("Query"):
        payload = {
            "user_query": user_query,
            "chunks": chunks,
            "rag_type": rag_type,
            "top_k": top_k
        }
        if use_gemini:
            with st.spinner("RAG Retrieval + Gemini Generation..."):
                try:
                    resp = requests.post(f"{base_url}/rag-ask-gemini", json=payload)
                    resp.raise_for_status()
                    answer = resp.json().get("answer", "")
                    st.write("### Final answer from Gemini:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Failed to call /rag-ask-gemini: {e}")
        else:
            with st.spinner("RAG Retrieval..."):
                try:
                    resp = requests.post(f"{base_url}/rag-query", json=payload)
                    resp.raise_for_status()
                    result_data = resp.json()
                    top_chunks = result_data.get("top_chunks", [])
                    if not top_chunks:
                        st.warning("No relevant chunks retrieved.")
                    else:
                        st.write("### Retrieved text chunks by RAG:")
                        for i, c in enumerate(top_chunks):
                            st.markdown(f"**Result {i+1}**:\n```\n{c}\n```")
                except Exception as e:
                    st.error(f"Failed to call /rag-query: {e}")
