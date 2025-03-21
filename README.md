Access the services:
   - Airflow: http://localhost:8080 (username: airflow, password: airflow)
   - Streamlit UI: https://streamlit-frontend-service-1077531201115.us-east1.run.app
   - FastAPI: https://python-backend-service-1077531201115.us-east1.run.app
   - 
# RAG Pipeline with Airflow for NVIDIA Quarterly Reports

A Retrieval-Augmented Generation (RAG) pipeline for processing and querying NVIDIA quarterly reports using Apache Airflow for orchestration.

## Project Overview

This project implements a comprehensive RAG system that allows users to:
- Ingest and process PDF documents (NVIDIA quarterly reports)
- Parse PDFs using multiple strategies (basic, Docling, Mistral OCR)
- Generate text chunks using different chunking strategies
- Create vector embeddings and store them in various databases
- Query reports with specific filters like quarter/year
- Interact with the system through a user-friendly Streamlit interface

## Architecture

The project consists of two main pipelines:

1. **Data Pipeline (Airflow)**
   - Data ingestion (downloading NVIDIA reports)
   - PDF parsing and text extraction
   - Text chunking and embedding generation
   - Storage in vector databases

2. **Query Pipeline (FastAPI + Streamlit)**
   - User interface for uploading PDFs and sending queries
   - Backend API for processing requests
   - Integration with various RAG methods
   - LLM-powered response generation

## Components

- **Airflow**: Orchestrates the ETL processes
- **FastAPI**: Provides the backend API
- **Streamlit**: User interface for interacting with the system
- **Vector Databases**: Pinecone and ChromaDB for storing embeddings
- **LLM Integration**: Uses Mistral AI for generating responses

## Features

- **Multiple PDF parsing strategies**:
  - Basic extraction (PyMuPDF)
  - Docling-based parsing
  - Mistral OCR for improved extraction

- **Multiple chunking strategies**:
  - Fixed-size chunking
  - Paragraph-based chunking
  - Semantic chunking

- **Multiple RAG implementations**:
  - Manual embeddings with cosine similarity
  - Pinecone integration
  - ChromaDB integration

- **Hybrid search**:
  - Filter by specific quarters
  - Combine semantic and metadata-based filtering

## Setup and Installation

### Prerequisites
- Docker and Docker Compose
- Mistral AI API key
- Pinecone API key (for Pinecone integration)

### Environment Variables
Create a `.env` file in the project root with:

```
MISTRAL_API_KEY=your_mistral_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX=nvidia-reports
```

### Running the Project

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

2. Start the services:
```bash
docker-compose up -d
```



## Usage

### Data Ingestion
1. Access the Airflow UI
2. Trigger the `data_ingestion_dag` to download NVIDIA quarterly reports
3. Run the `pdf_parsing_dag` to parse the PDFs
4. Execute the `rag_indexing_dag` to create embeddings and store them

### Querying
1. Open the Streamlit UI
2. Select your preferred options:
   - Parser type
   - RAG method
   - Chunking strategy
   - Specific quarters (optional)
3. Enter your query
4. View the generated response and supporting context

### Uploading New PDFs
1. Go to the "Upload PDF" tab in the Streamlit UI
2. Select the PDF file, parser, chunking strategy, and quarter
3. Upload and process the document

## Project Structure

```
rag-pipeline/
├── airflow/
│   ├── dags/               # Airflow DAGs for orchestration
│   ├── plugins/            # Custom operators and hooks
│   └── Dockerfile          # Airflow service container
├── api/
│   ├── main.py             # FastAPI application
│   ├── routes/             # API endpoints
│   ├── services/           # Core business logic
│   └── Dockerfile          # API service container
├── ui/
│   ├── app.py              # Streamlit application
│   ├── pages/              # UI pages
│   └── Dockerfile          # UI service container
├── common/
│   ├── config.py           # Shared configuration
│   ├── models.py           # Shared data models
│   └── utils.py            # Utility functions
├── docker-compose.yml      # Docker services configuration
└── README.md               # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Ad
