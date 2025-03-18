import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_embeddings(chunks):
    """
    Compute embeddings for a list of text chunks
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of embeddings
    """
    # Extract texts from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = model.encode(texts)
    
    # Store embeddings with chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()
    
    return chunks

def compute_query_embedding(query_text):
    """
    Compute embedding for a query text
    
    Args:
        query_text: Query text
        
    Returns:
        Query embedding
    """
    # Generate embedding
    embedding = model.encode(query_text)
    
    return embedding

def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity score
    """
    # Convert to numpy arrays if they're not already
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    # Normalize the vectors
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    return np.dot(embedding1, embedding2)

def retrieve_similar_chunks(query_embedding, chunks, top_k=5):
    """
    Retrieve the most similar chunks to a query embedding
    
    Args:
        query_embedding: Query embedding
        chunks: List of chunks with embeddings
        top_k: Number of top results to return
        
    Returns:
        List of top_k most similar chunks
    """
    # Calculate similarity scores
    for chunk in chunks:
        chunk['score'] = cosine_similarity(query_embedding, chunk['embedding'])
    
    # Sort chunks by similarity score (descending)
    sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
    
    # Return top_k results
    return sorted_chunks[:top_k]