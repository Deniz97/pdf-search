.PHONY: build dev format typecheck db-up db-down db-reset migrate ingest reingest prune enrich reenrich chat test-questions search-eval download-pdfs

deps:
	uv sync

build:
	uv build

dev:
	uv run uvicorn app.main:app --reload

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

check: deps build format typecheck
	@echo "All is good"

db-up:
	docker compose up -d

db-down:
	docker compose down

db-reset:
	uv run python -c "import asyncio; from app.database import reset_db; asyncio.run(reset_db())"
	$(MAKE) migrate

migrate:
	uv run python -c "import asyncio; from app.database import init_db; asyncio.run(init_db())"

# Ingest targets: ROOT is an absolute path (default: $(CURDIR)/example-pdfs), N limits count (default: ALL)
ROOT ?= $(CURDIR)/example-pdfs
N ?= ALL

ingest:
	uv run python -m app.cli.ingest --path "$(abspath $(ROOT))" ingest $(N)

reingest:
	uv run python -m app.cli.ingest --path "$(abspath $(ROOT))" reingest $(N)

prune:
	uv run python -m app.cli.ingest prune

enrich:
	uv run python -m app.cli.enricher enrich $(N)

reenrich:
	uv run python -m app.cli.enricher reenrich $(N)

chat:
	uv run python -m app.cli.chat

# Search eval: generate test questions (per document), then run eval
N_QUESTIONS ?= 10

generate-questions:
	uv run python -m app.cli.generate_test_questions $(N_QUESTIONS)

# Alias for generate-questions (PHONY lists test-questions)
test-questions: generate-questions

search-eval:
	uv run python -m app.cli.run_search_eval

download-pdfs:
	uv run python scripts/download_pdfs.py
