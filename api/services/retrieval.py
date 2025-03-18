import os
import json
import pinecone
import chromadb
from chromadb.config import Settings
from api.services.embedding import compute_query_embedding, retrieve_similar_chunks

# Initialize Pinecone (if environment variables are set)
try:
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_env = os.environ.get("PINECONE_ENVIRONMENT")
    if pinecone_api_key and pinecone_env:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
except:
    print("Warning: Could not initialize Pinecone. Will use manual embeddings instead.")

# Initialize ChromaDB
chroma_dir = os.path.join("/app/data/indexes", "chromadb")
os.makedirs(chroma_dir, exist_ok=True)

try:
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=chroma_dir
    ))
except:
    print("Warning: Could not initialize ChromaDB. Will use manual embeddings instead.")
    chroma_client = None

def retrieve_documents(query, rag_method="chromadb", parser="docling", chunking_strategy="semantic", quarters=None, top_k=5):
    """
    Retrieve relevant document chunks based on the query using the specified retrieval method
    
    Args:
        query: User query string
        rag_method: Method to use for retrieval ("manual", "pinecone", "chromadb")
        parser: Parser used for documents
        chunking_strategy: Chunking strategy used
        quarters: List of quarters to filter by (e.g., ["2022_Q1", "2022_Q2"])
        top_k: Number of top results to return
        
    Returns:
        List of relevant document chunks
    """
    # Compute query embedding
    query_embedding = compute_query_embedding(query)
    
    # Prepare filter for quarters if provided
    quarter_filter = {}
    if quarters and len(quarters) > 0:
        quarter_filter = {"year_quarter": {"$in": quarters}}
    
    # Retrieve documents based on the specified method
    if rag_method == "manual":
        return retrieve_with_manual_embeddings(query_embedding, parser, chunking_strategy, quarters, top_k)
    elif rag_method == "pinecone":
        return retrieve_with_pinecone(query_embedding, parser, chunking_strategy, quarters, top_k)
    elif rag_method == "chromadb":
        return retrieve_with_chromadb(query, parser, chunking_strategy, quarters, top_k)
    else:
        # Default to manual embeddings
        return retrieve_with_manual_embeddings(query_embedding, parser, chunking_strategy, quarters, top_k)

def retrieve_with_manual_embeddings(query_embedding, parser, chunking_strategy, quarters, top_k=5):
    """
    Retrieve relevant document chunks using manual embeddings
    """
    # Load embeddings from file
    embeddings_dir = "/app/data/embeddings"
    embeddings_file = os.path.join(embeddings_dir, f"{parser}_embeddings.json")
    
    try:
        with open(embeddings_file, 'r') as f:
            chunks_with_embeddings = json.load(f)
    except:
        return []
    
    # Filter by chunking strategy
    filtered_chunks = [
        chunk for chunk in chunks_with_embeddings 
        if chunk['metadata']['chunk_strategy'] == chunking_strategy
    ]
    
    # Filter by quarters if provided
    if quarters and len(quarters) > 0:
        filtered_chunks = [
            chunk for chunk in filtered_chunks 
            if chunk['metadata']['year_quarter'] in quarters
        ]
    
    # Retrieve similar chunks
    return retrieve_similar_chunks(query_embedding, filtered_chunks, top_k)

def retrieve_with_pinecone(query_embedding, parser, chunking_strategy, quarters, top_k=5):
    """
    Retrieve relevant document chunks using Pinecone
    """
    try:
        # Get index name
        index_name = "nvidia-reports"  # Use a fixed index name from the DAG
        index = pinecone.Index(index_name)
        
        # Prepare filter for chunking strategy and quarters
        filter_dict = {"chunk_strategy": chunking_strategy}
        
        if quarters and len(quarters) > 0:
            filter_dict["year_quarter"] = {"$in": quarters}
        
        # Query Pinecone
        query_results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        results = []
        for match in query_results.matches:
            results.append({
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "metadata": match.metadata
            })
        
        return results
    except Exception as e:
        print(f"Error retrieving from Pinecone: {e}")
        # Fall back to manual embeddings
        return retrieve_with_manual_embeddings(query_embedding, parser, chunking_strategy, quarters, top_k)

def retrieve_with_chromadb(query, parser, chunking_strategy, quarters, top_k=5):
    """
    Retrieve relevant document chunks using ChromaDB
    """
    try:
        # Get collection
        collection_name = f"nvidia-{parser}"
        collection = chroma_client.get_collection(name=collection_name)
        
        # Prepare filter for chunking strategy and quarters
        where_clause = {"chunk_strategy": chunking_strategy}
        
        if quarters and len(quarters) > 0:
            where_clause["year_quarter"] = {"$in": quarters}
        
        # Query ChromaDB
        query_results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )
        
        # Format results
        results = []
        for i, doc_id in enumerate(query_results['ids'][0]):
            results.append({
                "id": doc_id,
                "text": query_results['documents'][0][i],
                "score": query_results['distances'][0][i] if 'distances' in query_results else 0.0,
                "metadata": query_results['metadatas'][0][i]
            })
        
        return results
    except Exception as e:
        print(f"Error retrieving from ChromaDB: {e}")
        # Fall back to manual embeddings
        query_embedding = compute_query_embedding(query)
        return retrieve_with_manual_embeddings(query_embedding, parser, chunking_strategy, quarters, top_k)