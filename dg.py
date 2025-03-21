from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.workflow import Airflow
from diagrams.onprem.client import Users
from diagrams.generic.storage import Storage
from diagrams.programming.language import Python
from diagrams.saas.chat import Slack
from diagrams.programming.framework import FastAPI
from diagrams.aws.database import Database

with Diagram("RAG Architecture", show=False, direction="LR"):

    # User Interface and Workflow Steps
    with Cluster("User Interface"):
        user = Users("Streamlit Frontend")
        upload_pdf = Python("Upload PDFs")
        select_parser = Python("Select PDF Parser")
        choose_rag = Python("Choose RAG Method")
        select_chunking = Python("Select Chunking Strategy")
        select_quarters = Python("Select Quarter(s)")
        submit_query = Python("Submit Query")

        user >> upload_pdf >> select_parser >> choose_rag >> select_chunking >> select_quarters >> submit_query

    # API Backend
    fastapi_backend = FastAPI("FastAPI Backend")
    submit_query >> Edge(label="Query & Options") >> fastapi_backend

    # Orchestration Layer
    airflow = Airflow("Airflow DAGs")

    # Storage Layer
    with Cluster("Storage Layer"):
        raw_storage = Storage("Raw PDF Storage")
        processed_storage = Storage("Processed Data\n(Quarterly)")

    # PDF Processing Pipeline
    with Cluster("PDF Processing"):
        basic_parser = Python("Basic Parser")
        docling_parser = Python("Docling Parser")
        mistral_ocr = Python("Mistral OCR")

        raw_storage >> [basic_parser, docling_parser, mistral_ocr] >> processed_storage

    # RAG Pipeline
    with Cluster("RAG Pipeline"):
        chunking_strategy = Python("Chunking Strategies")

        with Cluster("Vector Databases"):
            naive_rag = Python("Manual Embeddings")
            pinecone_db = Database("Pinecone")
            chromadb_db = Database("ChromaDB")

        chunking_strategy >> [naive_rag, pinecone_db, chromadb_db]

    # Hybrid Search Engine (Quarterly Data Filtering)
    hybrid_search = Python("Hybrid Search\n(Quarter Filter)")

    processed_storage >> hybrid_search
    [naive_rag, pinecone_db, chromadb_db] >> hybrid_search

    # LLM Integration
    llm = Slack("LLM (Mistral AI)")
    hybrid_search >> Edge(label="Contextual Chunks") >> llm

    # Response Flow and DAG Triggering
    fastapi_backend >> Edge(label="Trigger & Manage") >> airflow >> raw_storage
    fastapi_backend >> Edge(label="Process Query") >> hybrid_search
    llm >> Edge(label="Generated Response") >> fastapi_backend

    # Return Results to User Interface
    fastapi_backend >> Edge(label="Results") >> user
