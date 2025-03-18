import io
import os
import tempfile
import requests
from typing import Union, Dict, Any
import fitz  # PyMuPDF
from docling import Document  # Assuming docling is installed
import mistralai

# Environment variables for API keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

def parse_pdf_basic(pdf_content: bytes) -> str:
    """
    Basic PDF parsing using PyMuPDF
    """
    text = ""
    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_pdf_docling(pdf_content: bytes) -> str:
    """
    Parse PDF using Docling
    """
    # Save the PDF content to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_content)
        temp_path = temp_file.name
    
    try:
        # Use Docling to parse the PDF
        doc = Document.from_pdf(temp_path)
        text = doc.get_text()
        return text
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def parse_pdf_mistral_ocr(pdf_content: bytes) -> str:
    """
    Parse PDF using Mistral OCR API
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    client = mistralai.MistralClient(api_key=MISTRAL_API_KEY)
    
    # Save the PDF content to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_content)
        temp_path = temp_file.name
    
    try:
        # Use Mistral OCR to extract text
        with open(temp_path, "rb") as f:
            response = client.ocr(file=f)
        
        # Extract text from response
        return response.text
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def parse_pdf(pdf_content: bytes, parser_type: str = "docling") -> str:
    """
    Parse PDF using the specified parser type
    
    Args:
        pdf_content: PDF content as bytes
        parser_type: Type of parser to use (basic, docling, mistral_ocr)
        
    Returns:
        Extracted text from the PDF
    """
    parser_functions = {
        "basic": parse_pdf_basic,
        "docling": parse_pdf_docling,
        "mistral_ocr": parse_pdf_mistral_ocr
    }
    
    if parser_type not in parser_functions:
        raise ValueError(f"Unknown parser type: {parser_type}")
    
    return parser_functions[parser_type](pdf_content)