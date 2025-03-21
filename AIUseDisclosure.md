# AI Use Disclosure

This document outlines the AI tools and technologies used in the development of the NVIDIA Reports RAG System.

## AI Tools Used

### 1. Large Language Models (LLMs)

- **OpenAI GPT-3.5-turbo**: Used for query answering and content generation within the RAG system.
- **Claude 3.7 Sonnet**: Used to assist with code generation, architecture planning, and documentation.

### 2. Vector Embeddings and Retrieval

- **Sentence Transformers**: The project uses the "all-MiniLM-L6-v2" model for generating text embeddings.
- **Pinecone**: Used as a vector database for semantic search capabilities.
- **ChromaDB**: Used as an alternative vector database for retrieval.

### 3. OCR and Document Processing

- **Mistral OCR**: Used for advanced PDF text extraction.
- **Docling**: Employed for structured PDF parsing.

## Purpose and Implementation

### Document Understanding

AI is used to extract text from PDFs, preserving structure and semantics. Three different strategies are implemented:
1. Basic extraction (PyMuPDF)
2. Structure-aware extraction (Docling)
3. OCR-based extraction (Mistral OCR)

### Semantic Search

The project uses AI for semantic search in the following ways:
1. Embedding generation of text chunks
2. Similarity matching for retrieval
3. Vector database organization and search

### Query Answering

LLMs are used to process and generate human-readable answers based on retrieved context, implementing:
1. Context-aware response generation
2. Citation of sources
3. Highlighting relevant information

## Limitations and Considerations

- The RAG system's accuracy depends on the quality of document parsing and embeddings.
- LLM responses may sometimes hallucinate or provide incomplete information.
- The system is optimized for NVIDIA financial reports and may require adjustments for other document types.

## Development Support

AI assistance was used during development for:
- Code generation and debugging
- Architecture design
- Documentation creation
- Best practices implementation

All AI-generated code and content was reviewed and modified by human developers to ensure accuracy, efficiency, and proper integration.