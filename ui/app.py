import streamlit as st
import requests
import json
import os

# API endpoint
API_ENDPOINT = os.getenv("API_ENDPOINT", "http://api:8000")

st.set_page_config(
    page_title="NVIDIA Reports RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("NVIDIA Quarterly Reports RAG System")
st.markdown("""
This application allows you to query NVIDIA's quarterly reports using a Retrieval-Augmented Generation (RAG) system.
Upload PDFs or query existing documents with various parsing and retrieval options.
""")

# Sidebar for configuration
st.sidebar.title("Configuration")
parser_type = st.sidebar.selectbox(
    "PDF Parser",
    options=["docling", "mistral_ocr", "basic"],
    help="Select the PDF parsing method"
)

rag_method = st.sidebar.selectbox(
    "RAG Method",
    options=["manual", "pinecone", "chromadb"],
    help="Select the retrieval method"
)

chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    options=["fixed_size", "paragraph", "semantic"],
    help="Select the text chunking strategy"
)

# Main content area with tabs
tab1, tab2 = st.tabs(["Query Reports", "Upload PDF"])

# Tab 1: Query Reports
with tab1:
    st.header("Query NVIDIA Reports")
    
    # Quarter selection
    available_quarters = ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", 
                          "2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
                          "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
                          "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
                          "2024-Q1", "2024-Q2"]
    
    quarters = st.multiselect(
        "Select Quarters to Query",
        options=available_quarters,
        default=[],
        help="Leave empty to query all quarters"
    )
    
    query = st.text_area("Enter your query", height=100)
    
    if st.button("Submit Query"):
        if query:
            with st.spinner("Generating answer..."):
                try:
                    # Prepare request payload
                    payload = {
                        "query": query,
                        "parser_type": parser_type,
                        "rag_method": rag_method,
                        "chunking_strategy": chunking_strategy,
                        "quarters": quarters
                    }
                    
                    # Send request to API
                    response = requests.post(f"{API_ENDPOINT}/rag/query", json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Display result
                    st.subheader("Answer")
                    st.write(result["answer"])
                    
                    # Display context chunks
                    with st.expander("View Source Context"):
                        for i, chunk in enumerate(result["context_chunks"]):
                            st.markdown(f"**Chunk {i+1}**")
                            st.text(chunk)
                            st.divider()
                    
                    # Display metadata
                    with st.expander("View Metadata"):
                        st.json(result["metadata"])
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Tab 2: Upload PDF
with tab2:
    st.header("Upload NVIDIA Report")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    quarter = st.selectbox("Select Quarter", options=available_quarters)
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    # Prepare form data
                    files = {"file": uploaded_file.getvalue()}
                    data = {
                        "parser_type": parser_type,
                        "chunking_strategy": chunking_strategy,
                        "quarter": quarter
                    }
                    
                    # Send request to API
                    response = requests.post(
                        f"{API_ENDPOINT}/rag/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        data=data
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Display result
                    st.success(f"PDF processed successfully! {result['chunks_count']} chunks created.")
                    
                    # Display metadata
                    with st.expander("View Processing Details"):
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("NVIDIA RAG Pipeline © 2025")