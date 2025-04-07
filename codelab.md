---

id: docs

title: "RAG Pipeline with Airflow for NVIDIA Quarterly Reports"

summary: "A codelab for building a Retrieval-Augmented Generation (RAG) pipeline to process and query NVIDIA quarterly reports using Apache Airflow, FastAPI, and Streamlit."

authors: ["Group 1"]

categories: ["AI", "Data Pipelines", "Document Processing"]

environments: ["Web", "Cloud"]

status: "Published"

---


# RAG Pipeline with Airflow for NVIDIA Quarterly Reports

## Introduction
This codelab walks you through building a **Retrieval-Augmented Generation (RAG) pipeline** for processing and querying NVIDIA quarterly reports. The project uses **Apache Airflow** for orchestration, **Streamlit** for the user interface, and **FastAPI** for backend services. You'll also integrate vector databases like Pinecone and ChromaDB to optimize retrieval.

By the end of this codelab, you will:
- Set up the project environment.
- Implement data ingestion and processing pipelines.
- Parse PDFs using multiple strategies.
- Build a RAG system with hybrid search capabilities.
- Deploy the system using Docker.

---

## Step 1: Setup and Installation
### **Objective**
In this step, we will set up the environment and prepare the necessary tools to start building the RAG pipeline.

### **Prerequisites**
Before proceeding, ensure you have the following installed on your system:
- **Docker**: For containerizing services.
- **Docker Compose**: To manage multi-container setups.
- **Python 3.9+**: Required for running scripts and services.
- **Git**: To clone the repository.
- **Mistral AI API Key**: For OCR-based text extraction.
- **Pinecone API Key**: For vector database integration.

### **Environment Variables**
Create a `.env` file in the project root directory with the following variables:
MISTRAL_API_KEY=your_mistral_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX=nvidia-reports

text

### **Setting Up the Project**
1. Clone the repository:
git clone https://github.com/DAMG-7245/assignment_4-2.git




2. Install Python dependencies:
pip install -r requirements.txt

text

3. Verify that Docker is installed:
docker --version
docker-compose --version

text

4. Ensure that your `.env` file is correctly configured.

S3_BUCKET_NAME=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
PINECONE_API_KEY = 
PINECONE_ENVIRONMENT = 

---

## Step 2: Data Ingestion Pipeline

### Overview
The first step in building the RAG pipeline is ingesting NVIDIA's quarterly reports. This involves:
1. Downloading PDF files from NVIDIA's investor relations website.
2. Storing them in a designated folder for further processing.

### Airflow DAG for Data Ingestion
The data ingestion process is orchestrated using an Airflow DAG. Below is a placeholder for defining the DAG:

Placeholder: Define Airflow DAG for data ingestion
Explaination:
- This DAG downloads NVIDIA quarterly reports from a specified URL.
- It stores them in a "raw" folder within the data/ directory.
text

---

## Step 3: Parsing PDFs

### Overview
After downloading the reports, parse them to extract meaningful text. This step implements three parsing strategies:
1. Basic extraction using libraries like PyMuPDF.
2. Advanced parsing with Docling.
3. OCR-based parsing using Mistral OCR.

### Parsing Strategies Implementation
Below is a placeholder for implementing PDF parsing strategies:

Placeholder: Implement PDF parsing strategies
Explaination:
- Basic extraction uses PyMuPDF to extract raw text from PDFs.
- Docling provides advanced parsing capabilities for structured data extraction.
- Mistral OCR handles scanned PDFs or complex layouts effectively.
text

---

## Step 4: Chunking and Embedding Generation

### Overview
Once text is extracted, split it into smaller chunks to optimize retrieval. This step covers:
1. Fixed-size chunking.
2. Paragraph-based chunking.
3. Semantic chunking.

Generate vector embeddings for each chunk using models like Hugging Face Transformers.

### Chunking Implementation
Below is a placeholder for chunking strategies:

Placeholder: Implement chunking strategies
Explaination:
- Fixed-size chunks split text into predefined lengths (e.g., 500 words).
- Paragraph-based chunks split text based on natural paragraph boundaries.
- Semantic chunks split text based on meaning or context relevance.
Placeholder: Generate embeddings using Hugging Face Transformers
Explaination:
- Use pre-trained models like BERT or Sentence Transformers to generate embeddings.
text

---

## Step 5: Building the RAG Pipeline

### Overview
The RAG pipeline retrieves relevant document chunks based on user queries and generates responses using an LLM. This step involves:
1. Implementing manual cosine similarity-based retrieval.
2. Integrating Pinecone and ChromaDB for advanced retrieval.
3. Combining semantic filtering with metadata filtering (hybrid search).

### RAG Pipeline Implementation
Below is a placeholder for RAG pipeline implementation:

Placeholder: Implement naive RAG system with cosine similarity
Explaination:
- Compute embeddings manually and calculate cosine similarity between query and document chunks.
Placeholder: Integrate Pinecone and ChromaDB
Explaination:
- Store embeddings in vector databases for efficient retrieval at scale.
Placeholder: Implement hybrid search
Explaination:
- Combine semantic similarity with metadata filtering (e.g., by quarter/year).
text

---

## Step 6: User Interface with Streamlit

### Overview
The Streamlit application allows users to interact with the system by uploading PDFs, selecting parsers, chunking strategies, and querying specific quarters' data.

### Streamlit Application Implementation
Below is a placeholder for Streamlit application code:

Placeholder: Build Streamlit application
Explaination:
- The app provides an intuitive UI where users can upload PDFs, select options, and view query results.
text

---

## Step 7: Deployment

### Overview
Deploy all components using Docker containers. This includes:
1. Airflow pipeline container.
2. Streamlit + FastAPI container.

### Docker Deployment Configuration
Below is a placeholder for Docker deployment configuration:

Placeholder: docker-compose.yml configuration
Explaination:
- Defines services for Airflow, Streamlit, and FastAPI.
- Ensures all components run seamlessly in isolated containers.
text

---

## Conclusion

Congratulations! You have successfully built a RAG pipeline for processing NVIDIA quarterly reports using Apache Airflow, Streamlit, FastAPI, and vector databases like Pinecone/ChromaDB.

Feel free to extend this project by adding new features or optimizing existing ones!

---
