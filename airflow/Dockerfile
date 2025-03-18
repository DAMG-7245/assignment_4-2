FROM apache/airflow:2.7.1

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install Docling and other dependencies
RUN pip install --no-cache-dir docling pinecone-client chromadb langchain sentence-transformers openai tiktoken pypdf mistralai