# pdf-search - FastAPI app with OCR support
FROM python:3.11-slim-bookworm

# System deps: poppler (pdf2image), tesseract (pytesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Create minimal app package so uv can resolve project deps
RUN mkdir -p app && touch app/__init__.py

# Install deps (no dev, no project - we copy real app after)
RUN uv sync --frozen --no-dev --no-install-project

# Copy app code
COPY app ./app
COPY templates ./templates
COPY static ./static

# Expose app port
EXPOSE 9287

# Run uvicorn (bind 0.0.0.0 for external access)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9287"]
