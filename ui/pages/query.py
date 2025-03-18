import streamlit as st
import requests
import os

def show():
    st.title("Query NVIDIA Reports")
    
    # API endpoint
    API_ENDPOINT = os.getenv("API_ENDPOINT", "http://api:8000")
    
    # Sidebar config
    st.sidebar.title("Retrieval Settings")
    
    rag_method = st.sidebar.selectbox(
        "RAG Method",
        options=["manual", "pinecone", "chromadb"],
        help="Select the retrieval method"
    )
    
    chunking_strategy = st.sidebar.selectbox(
        "Chunking Strategy",
        options=["fixed_size", "paragraph", "semantic"],
        help="Select the text chunking strategy that was used"
    )
    
    parser_type = st.sidebar.selectbox(
        "Original Parser",
        options=["docling", "mistral_ocr", "basic"],
        help="Select the PDF parsing method that was used"
    )
    
    # Quarter selection
    available_quarters = ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", 
                         "2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
                         "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
                         "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
                         "2024-Q1", "2024-Q2"]
    
    st.write("Select specific quarters to narrow your search:")
    quarters = st.multiselect(
        "Quarters",
        options=available_quarters,
        default=[],
        help="Leave empty to query all quarters"
    )
    
    # Query input
    st.write("Enter your question about NVIDIA quarterly reports:")
    query = st.text_area("Query", height=100)
    
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