# PDF Search API

A FastAPI service that lets you upload PDF documents and search through them using natural language queries. Uses PostgreSQL with pgvector for vector storage, OpenAI embeddings for semantic search, and OCR (pdf2image + pytesseract) for text extraction.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/) (for PostgreSQL + pgvector)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` on macOS)
- [Poppler](https://poppler.freedesktop.org/) (`brew install poppler` on macOS — required by pdf2image)
- An OpenAI API key

## Quick Start

```bash
# 1. Start PostgreSQL with pgvector
docker compose up -d

# 2. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 3. Install dependencies
uv sync

# 4. Run the server
uv run uvicorn app.main:app --reload
```

The server will be available at `http://localhost:8000`:
- **Web UI**: http://localhost:8000 (Search page) and http://localhost:8000/chat (Chat page)
- **API Docs**: http://localhost:8000/docs
- **API**: http://localhost:8000/api/v1

## Web Interface

The application includes a web UI built with Jinja2 templates and HTMX:

- **Search Page** (`/`) - Query documents and view matched sections with scores
- **Chat Page** (`/chat`) - Interactive chat interface with tool calling support

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Search page (web UI) |
| `GET`  | `/chat` | Chat page (web UI) |
| `POST` | `/api/v1/documents/upload` | Upload a PDF file for processing |
| `POST` | `/api/v1/documents/query` | Search documents with natural language (returns LLM answer) |
| `POST` | `/api/v1/documents/search` | Search documents (returns matched chunks only) |
| `POST` | `/api/v1/chat` | Chat endpoint with tool calling |
| `GET`  | `/api/v1/documents` | List all uploaded documents |
| `GET`  | `/health` | Health check |

### Upload a PDF

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@sample.pdf"
```

### Query documents

```bash
curl -X POST http://localhost:8000/api/v1/documents/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5}'
```

## Architecture

1. **PDF Upload**: PDF is converted to images (pdf2image), then OCR'd (pytesseract)
2. **Chunking**: Text is split into chunks of 5 sentences with 1 sentence overlap
3. **Embedding**: Each chunk is embedded using OpenAI's `text-embedding-3-small`
4. **Storage**: Chunks and embeddings are stored in PostgreSQL with pgvector
5. **Query**: User query is embedded, nearest chunks are retrieved via cosine distance, and an LLM generates an answer from the context
