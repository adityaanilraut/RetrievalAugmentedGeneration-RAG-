# Sample knowledge base

## Project

This is a **test document** for the RAG system. The retrieval pipeline chunks text, embeds it with OpenAI, and stores vectors in PostgreSQL using the `pgvector` extension.

## Facts

- The API exposes `/health`, `/ingest`, and `/query` endpoints.
- Citations in answers refer to source labels like `[S1]` tied to chunk IDs.
