import streamlit as st
import requests
import os

def show():
    st.title("Upload NVIDIA Report")
    
    # API endpoint
    API_ENDPOINT = os.getenv("API_ENDPOINT", "http://api:8000")
    
    # PDF parser selection
    parser_type = st.selectbox(
        "Select PDF Parser",
        options=["docling", "mistral_ocr", "basic"],
        help="Choose the method to extract text from PDFs"
    )
    
    # Chunking strategy selection
    chunking_strategy = st.selectbox(
        "Select Chunking Strategy",
        options=["fixed_size", "paragraph", "semantic"],
        help="Choose how to split the text into chunks"
    )
    
    # Quarter selection
    available_quarters = ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", 
                         "2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
                         "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
                         "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
                         "2024-Q1", "2024-Q2"]
    
    quarter = st.selectbox("Select Quarter", options=available_quarters)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    # Prepare form data
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