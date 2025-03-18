import os
from typing import List, Dict, Any
import mistralai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Environment variables for API keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

def generate_response(query: str, context_chunks: List[str]) -> str:
    """
    Generate a response using an LLM based on the query and context chunks
    
    Args:
        query: User query
        context_chunks: Retrieved context chunks
        
    Returns:
        Generated response
    """
    # Initialize Mistral client
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    client = MistralClient(api_key=MISTRAL_API_KEY)
    
    # Format context for the LLM
    formatted_context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
    
    # Construct the prompt
    system_prompt = """You are an AI assistant tasked with answering questions about NVIDIA quarterly reports.
Use only the provided context to answer the query. If the information is not in the context, say you don't know.
Cite specific quarters when possible."""
    
    user_prompt = f"""Context:
{formatted_context}

Query: {query}

Please provide a comprehensive answer based only on the information in the context."""
    
    # Generate response
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt)
    ]
    
    response = client.chat(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.1,
        max_tokens=1000
    )
    
    return response.choices[0].message.content