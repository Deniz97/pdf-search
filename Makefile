.PHONY: build dev format typecheck db-up db-down migrate ingest chat

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

migrate:
	uv run python -c "import asyncio; from app.database import init_db; asyncio.run(init_db())"

ingest:
	uv run python -m app.ingest

chat:
	uv run python -m app.chat
