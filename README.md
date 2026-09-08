# AI Document Assistant using RAG

A production-grade Retrieval-Augmented Generation (RAG) system for document intelligence, semantic question answering, multi-document comparison, hierarchical summarization, strategic insight extraction, and action-item detection.

## System Architecture

```text
                 ┌──────────────┐
                 │    User      │
                 └──────┬───────┘
                        │ REST API / JWT
                        ▼
                 ┌──────────────┐
                 │   FastAPI    │
                 │   Backend    │
                 └──────┬───────┘
                        │
             ┌──────────┼───────────┐
             │          │           │
             ▼          ▼           ▼
        SQLAlchemy   OpenAI      Document
        Database      API        Parsers
        (SQLite/   (Embeddings/  (PyMuPDF/
        pgvector)   Responses)    python-docx)
             │
             ▼
       Hybrid Search
   (Vector + Keyword)
             │
             ▼
      Grounded Answer +
      Source Citations
```

## Features

- **Multi-Format Ingestion**: Supports PDF, DOCX, and TXT files with document structure and page tracking.
- **Overlapping Semantic Chunking**: Sliding window token-aware chunker preserving page numbers, section indices, and document IDs.
- **OpenAI Embedding Pipeline**: `text-embedding-3-small` vector embedding generator with batch processing.
- **Hybrid Similarity Search**: Combines Cosine Vector Similarity with Keyword TF-IDF matching and confidence threshold filtering.
- **Prompt Injection Defense**: Sanitizes untrusted document text and enforces evidence-grounded system prompts.
- **Document Summarization**: Hierarchical map-reduce summarizer for large multi-page files.
- **Structured Extraction**: Extract key insights and action items with owners, deadlines, and priorities.
- **Multi-Turn Context & Query Rewriting**: Maintains conversation memory and rephrases follow-up queries.
- **RAG Evaluation Suite**: Benchmark measurement for Precision@K, Recall@K, MRR, Groundedness, and Hallucination rate.
- **JWT Authentication**: User isolation ensuring strict data privacy.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=sqlite:///./sql_app.db
   JWT_SECRET=super_secret_jwt_key_change_in_production
   ```

3. Run backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. Run unit tests:
   ```bash
   python test_rag.py
   ```
