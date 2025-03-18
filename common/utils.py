import os
import json
import logging
import hashlib
import re
from typing import Dict, List, Any, Union, Optional, Tuple
import boto3
from botocore.exceptions import ClientError
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_id(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a unique ID for a text chunk based on its content and metadata
    
    Args:
        text: The text content
        metadata: Optional metadata dictionary
    
    Returns:
        A unique ID string
    """
    content = text
    if metadata:
        content += json.dumps(metadata, sort_keys=True)
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def upload_to_s3(file_path: str, bucket: str, object_name: Optional[str] = None) -> bool:
    """
    Upload a file to an S3 bucket
    
    Args:
        file_path: File to upload
        bucket: Bucket to upload to
        object_name: S3 object name. If not specified then file_name is used
    
    Returns:
        True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_path)
        
    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket, object_name)
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        return False
    return True

def download_from_s3(bucket: str, object_name: str, file_path: str) -> bool:
    """
    Download a file from an S3 bucket
    
    Args:
        bucket: Bucket to download from
        object_name: S3 object name
        file_path: Local path to save the file
    
    Returns:
        True if file was downloaded, else False
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket, object_name, file_path)
    except ClientError as e:
        logger.error(f"S3 download error: {e}")
        return False
    return True

def extract_year_quarter(text: str) -> Optional[str]:
    """
    Extract year and quarter from text using regex
    
    Args:
        text: Text to extract from
    
    Returns:
        Formatted quarter string (e.g., "2023-Q1") or None if not found
    """
    # Look for patterns like "Q1 2023", "2023 Q1", "first quarter 2023", etc.
    pattern1 = r"Q([1-4])\s*(\d{4})"
    pattern2 = r"(\d{4})\s*Q([1-4])"
    pattern3 = r"(first|second|third|fourth)\s+quarter\s+(\d{4})"
    
    # Try matching patterns
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        quarter, year = match.groups()
        return f"{year}-Q{quarter}"
    
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        year, quarter = match.groups()
        return f"{year}-Q{quarter}"
    
    match = re.search(pattern3, text, re.IGNORECASE)
    if match:
        quarter_text, year = match.groups()
        quarter_map = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
        quarter = quarter_map.get(quarter_text.lower())
        return f"{year}-Q{quarter}"
    
    return None

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep periods, commas, etc.
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text.strip()

def chunk_overlap_percentage(chunk1: str, chunk2: str) -> float:
    """
    Calculate the percentage of overlap between two text chunks
    
    Args:
        chunk1: First text chunk
        chunk2: Second text chunk
    
    Returns:
        Overlap percentage (0-100)
    """
    # Tokenize by word
    tokens1 = set(chunk1.lower().split())
    tokens2 = set(chunk2.lower().split())
    
    # Calculate overlap
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union) * 100.0

def iso_date_string() -> str:
    """
    Get current date-time as ISO format string
    
    Returns:
        ISO format date-time string
    """
    return datetime.now().isoformat()