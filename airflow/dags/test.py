from datetime import datetime, timedelta
import os
import json
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from sentence_transformers import SentenceTransformer
import chromadb
import pinecone
from chromadb.config import Settings
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.utils import openai_token_count

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

PROCESSED_DATA_DIR = "/opt/airflow/data/processed"
EMBEDDINGS_DIR = "/opt/airflow/data/embeddings"
INDEX_DIR = "/opt/airflow/data/indexes"
CHUNK_DIR = "/opt/airflow/data/chunks"

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunking Strategies

def chunk_by_paragraph(text):
    """Chunk text by paragraphs"""
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Further split very long paragraphs
    chunks = []
    for para in paragraphs:
        if len(para) > 1500:
            chunks.extend(chunk_by_fixed_size(para, 1000, 150))
        else:
            chunks.append(para)
            
    return chunks

def chunk_by_recursive_token(text):
    """Chunk text using RecursiveTokenChunker"""
    # Initialize the chunker with our configured settings
    chunker = RecursiveTokenChunker(
        chunk_size=400,
        chunk_overlap=50,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    
    # Split the text into chunks
    chunks = chunker.split_text(text)
    
    return chunks

def chunk_by_kaamradt(text):
    """
    Implement Kaamradt chunking strategy
    This strategy focuses on semantic boundaries and hierarchical structure
    """
    # First identify potential section boundaries
    lines = text.split('\n')
    sections = []
    current_section = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Identify headers and section markers
        is_header = (line.isupper() or 
                    (len(line) < 80 and (line.endswith(':') or line.endswith('.'))) or
                    any(marker in line for marker in ['SECTION', 'CHAPTER', 'PART']))
        
        if is_header and current_section and len('\n'.join(current_section)) > 100:
            # End current section and start a new one
            sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    # Add the final section
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Process each section to ensure appropriate chunk size
    chunks = []
    for section in sections:
        # For very short sections, combine them
        if len(section) < 300 and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + section
        # For very long sections, split them further
        elif len(section) > 1500:
            # Use recursive token chunker for long sections
            sub_chunks = chunk_by_recursive_token(section)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)
    
    return chunks

def chunk_by_fixed_size(text, chunk_size=1000, overlap=200):
    """Helper function for fixed size chunking"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Find the nearest period or newline to make chunks end naturally
        if end < len(text):
            for marker in ['. ', '.\n', '\n\n', '\n', ' ']:
                natural_end = text.rfind(marker, start, end)
                if natural_end != -1:
                    end = natural_end + len(marker)
                    break
        
        chunks.append(text[start:end].strip())
        start = max(start, end - overlap)
        
        # Avoid infinite loop for small texts
        if start >= len(text) or end == len(text):
            break
            
    return chunks

# Process documents for embedding and indexing

def process_documents(parser_type, **kwargs):
    """
    Process all documents parsed with the specified parser
    and prepare them for different RAG strategies
    """
    parser_dir = os.path.join(PROCESSED_DATA_DIR, parser_type)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)
    
    # Get all parsed JSON files
    parsed_files = []
    for file in os.listdir(parser_dir):
        if file.endswith('.json'):
            parsed_files.append(os.path.join(parser_dir, file))
    
    all_chunks = []
    
    for file_path in parsed_files:
        with open(file_path, 'r') as f:
            parsed_data = json.load(f)
        
        # Extract document metadata
        doc_id = os.path.basename(file_path).replace('.json', '')
        year_quarter = parsed_data.get('year_quarter', 'unknown')
        
        # Concatenate all text from the document
        full_text = ' '.join([page['text'] for page in parsed_data['content']])
        
        # Apply different chunking strategies
        paragraph_chunks = chunk_by_paragraph(full_text)
        recursive_chunks = chunk_by_recursive_token(full_text)
        kaamradt_chunks = chunk_by_kaamradt(full_text)
        
        # Store all chunks with metadata
        for i, chunk in enumerate(paragraph_chunks):
            all_chunks.append({
                'id': f"{doc_id}_para_{i}",
                'text': chunk,
                'doc_id': doc_id,
                'year_quarter': year_quarter,
                'chunk_strategy': 'paragraph',
                'chunk_index': i,
                'parser': parser_type
            })
            
        for i, chunk in enumerate(recursive_chunks):
            all_chunks.append({
                'id': f"{doc_id}_recursive_{i}",
                'text': chunk,
                'doc_id': doc_id,
                'year_quarter': year_quarter,
                'chunk_strategy': 'recursive',
                'chunk_index': i,
                'parser': parser_type
            })
            
        for i, chunk in enumerate(kaamradt_chunks):
            all_chunks.append({
                'id': f"{doc_id}_kaamradt_{i}",
                'text': chunk,
                'doc_id': doc_id,
                'year_quarter': year_quarter,
                'chunk_strategy': 'kaamradt',
                'chunk_index': i,
                'parser': parser_type
            })
    
    # Save all chunks to a JSON file
    output_path = os.path.join(CHUNK_DIR, f"{parser_type}_chunks.json")
    with open(output_path, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    return output_path

def create_naive_embeddings(chunk_file, **kwargs):
    """Create embeddings for naive RAG (no vector DB)"""
    with open(chunk_file, 'r') as f:
        chunks = json.load(f)
    
    # Extract texts and metadata
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = model.encode(texts)
    
    # Store embeddings with metadata
    embedding_data = []
    for i, chunk in enumerate(chunks):
        embedding_data.append({
            'id': chunk['id'],
            'text': chunk['text'],
            'embedding': embeddings[i].tolist(),
            'metadata': {
                'doc_id': chunk['doc_id'],
                'year_quarter': chunk['year_quarter'],
                'chunk_strategy': chunk['chunk_strategy'],
                'chunk_index': chunk['chunk_index'],
                'parser': chunk['parser']
            }
        })
    
    # Save embeddings to a file
    parser_type = os.path.basename(chunk_file).replace('_chunks.json', '')
    output_path = os.path.join(EMBEDDINGS_DIR, f"{parser_type}_embeddings.json")
    with open(output_path, 'w') as f:
        json.dump(embedding_data, f, indent=2)
    
    return output_path

def index_with_chromadb(chunk_file, **kwargs):
    """Index chunks using ChromaDB vector database"""
    with open(chunk_file, 'r') as f:
        chunks = json.load(f)
    
    # Extract texts and metadata
    texts = [chunk['text'] for chunk in chunks]
    ids = [chunk['id'] for chunk in chunks]
    
    metadata_list = [{
        'doc_id': chunk['doc_id'],
        'year_quarter': chunk['year_quarter'],
        'chunk_strategy': chunk['chunk_strategy'],
        'chunk_index': chunk['chunk_index'],
        'parser': chunk['parser']
    } for chunk in chunks]
    
    # Initialize ChromaDB
    chroma_dir = os.path.join(INDEX_DIR, "chromadb")
    os.makedirs(chroma_dir, exist_ok=True)
    
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=chroma_dir
    ))
    
    # Create or get collection
    parser_type = os.path.basename(chunk_file).replace('_chunks.json', '')
    collection_name = f"nvidia-{parser_type}"
    
    # Check if collection exists, if not create it
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)
    
    # Add documents in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_metadata = metadata_list[i:i+batch_size]
        
        # Add to ChromaDB
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadata
        )
    
    # Persist the database
    client.persist()
    
    # Save index information
    output_path = os.path.join(INDEX_DIR, f"{parser_type}_chromadb.json")
    with open(output_path, 'w') as f:
        json.dump({
            'collection_name': collection_name,
            'count': len(chunks),
            'parser': parser_type
        }, f, indent=2)
    
    return output_path

def index_with_pinecone(chunk_file, **kwargs):
    """Index chunks using Pinecone vector database"""
    with open(chunk_file, 'r') as f:
        chunks = json.load(f)
    
    # Extract texts and metadata
    texts = [chunk['text'] for chunk in chunks]
    metadata_list = [{
        'doc_id': chunk['doc_id'],
        'year_quarter': chunk['year_quarter'],
        'chunk_strategy': chunk['chunk_strategy'],
        'chunk_index': chunk['chunk_index'],
        'parser': chunk['parser']
    } for chunk in chunks]
    
    # Generate embeddings
    embeddings = model.encode(texts)
    
    # Initialize Pinecone
    pinecone.init(
        api_key=Variable.get("PINECONE_API_KEY"),
        environment=Variable.get("PINECONE_ENVIRONMENT")
    )
    
    # Get or create index
    index_name = "nvidia-reports"
    dimension = len(embeddings[0])
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"
        )
    
    # Get the index
    index = pinecone.Index(index_name)
    
    # Upsert data in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = [chunk['id'] for chunk in chunks[i:i+batch_size]]
        batch_embeddings = [embedding.tolist() for embedding in embeddings[i:i+batch_size]]
        batch_metadata = metadata_list[i:i+batch_size]
        
        # Create (id, vector, metadata) tuples
        vectors_batch = list(zip(batch_ids, batch_embeddings, batch_metadata))
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors_batch)
    
    # Save index information
    parser_type = os.path.basename(chunk_file).replace('_chunks.json', '')
    output_path = os.path.join(INDEX_DIR, f"{parser_type}_pinecone.json")
    with open(output_path, 'w') as f:
        json.dump({
            'index_name': index_name,
            'dimension': dimension,
            'count': len(chunks),
            'parser': parser_type
        }, f, indent=2)
    
    return output_path


with DAG(
    'rag_indexing',
    default_args=default_args,
    description='Process parsed PDFs for RAG indexing with three chunking strategies',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 3, 1),
    catchup=False,
    tags=['nvidia', 'rag', 'indexing'],
) as dag:
    
    # Create task groups for each parser type
    parser_types = ['pypdf', 'docling', 'mistral_ocr']
    
    for parser_type in parser_types:
        with TaskGroup(group_id=f'process_{parser_type}') as parser_group:
            # Process documents into chunks
            process_task = PythonOperator(
                task_id=f'process_{parser_type}_docs',
                python_callable=process_documents,
                op_kwargs={'parser_type': parser_type},
            )
            
            # Create naive embeddings
            naive_task = PythonOperator(
                task_id=f'naive_embeddings_{parser_type}',
                python_callable=create_naive_embeddings,
                op_kwargs={'chunk_file': process_task.output},
            )
            
            # Index with ChromaDB
            chromadb_task = PythonOperator(
                task_id=f'chromadb_index_{parser_type}',
                python_callable=index_with_chromadb,
                op_kwargs={'chunk_file': process_task.output},
            )
            # Inside the parser_group TaskGroup:
            pinecone_task = PythonOperator(
                task_id=f'pinecone_index_{parser_type}',
                python_callable=index_with_pinecone,
                op_kwargs={'chunk_file': process_task.output},
            )

            # UTask dependencies:
            process_task >> [naive_task, chromadb_task, pinecone_task]

          