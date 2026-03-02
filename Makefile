.PHONY: build dev format typecheck db-up db-down db-reset migrate ingest reingest prune enrich reenrich chat search-eval download-pdfs kill-dev unlock-documents

deps:
	uv sync

build:
	uv build

dev:
	uv run uvicorn app.main:app --reload

kill-dev:
	@-kill -9 $$(lsof -t -i :8000) 2>/dev/null; true

unlock-documents:
	uv run python scripts/unlock_documents.py

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

check: deps build format typecheck
	@echo "All is good"

# Docker Compose (app only; DB is external)
dc-up-build:
	docker compose up -d --build

dc-logs:
	docker compose logs -f

db-reset:
	uv run python -c "import asyncio; from app.database import reset_db; asyncio.run(reset_db())"
	$(MAKE) migrate

migrate:
	$(MAKE) unlock-documents
	uv run python -c "import asyncio; from app.database import init_db; asyncio.run(init_db())"
	uv run python scripts/migrate_search_eval_results.py

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
	uv run python -m app.cli.enricher enrich $(N) --workers 4

reenrich:
	uv run python -m app.cli.enricher reenrich $(N) --workers 4

chat:
	uv run python -m app.cli.chat

# Search eval: generate test questions (per document), then run eval
N_QUESTIONS ?= 10

generate-questions:
	uv run python -m app.cli.generate_test_questions --questions-per-document $(N_QUESTIONS) --workers 4

# Optional: RESUME_RUN_ID=<uuid> to resume interrupted run
search-eval:
	uv run python -m app.cli.run_search_eval --documents 25 --questions-per-document 4 --workers 4 $(if $(RESUME_RUN_ID),--resume-run-id $(RESUME_RUN_ID),)

download-pdfs:
	uv run python scripts/download_pdfs.py --limit
