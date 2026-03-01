"""CLI chatbot with RAG tool calling."""

import json
import sys

from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.embeddings import get_embedding

# ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

engine = create_engine(settings.database_url_sync)
client = OpenAI(api_key=settings.openai_api_key)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search across ALL uploaded documents for chunks semantically similar "
                "to the query. Returns the most relevant text passages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural language search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_book",
            "description": (
                "Search within a SPECIFIC document (by filename) for chunks semantically "
                "similar to the query. Use this when the user asks about a specific PDF."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_name": {
                        "type": "string",
                        "description": "The exact filename of the PDF document to search in",
                    },
                    "query": {
                        "type": "string",
                        "description": "The natural language search query",
                    },
                },
                "required": ["book_name", "query"],
            },
        },
    },
]


def get_document_list() -> list[dict]:
    with Session(engine) as session:
        rows = session.execute(
            text("SELECT filename, page_count FROM documents ORDER BY filename")
        ).fetchall()
    return [{"filename": r.filename, "page_count": r.page_count} for r in rows]


def do_search(query: str, top_k: int = 8) -> list[dict]:
    embedding = get_embedding(query)
    with Session(engine) as session:
        rows = session.execute(
            text("""
                SELECT c.content, c.page_number, c.chunk_index,
                       d.filename,
                       c.embedding <=> :embedding AS distance
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> :embedding
                LIMIT :limit
            """),
            {"embedding": str(embedding), "limit": top_k},
        ).fetchall()
    return [
        {
            "document": r.filename,
            "page": r.page_number,
            "chunk_index": r.chunk_index,
            "score": round(1 - r.distance, 4),
            "content": r.content,
        }
        for r in rows
    ]


def do_search_in_book(book_name: str, query: str, top_k: int = 8) -> list[dict]:
    embedding = get_embedding(query)
    with Session(engine) as session:
        rows = session.execute(
            text("""
                SELECT c.content, c.page_number, c.chunk_index,
                       d.filename,
                       c.embedding <=> :embedding AS distance
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.filename = :book_name
                ORDER BY c.embedding <=> :embedding
                LIMIT :limit
            """),
            {"embedding": str(embedding), "book_name": book_name, "limit": top_k},
        ).fetchall()
    return [
        {
            "document": r.filename,
            "page": r.page_number,
            "chunk_index": r.chunk_index,
            "score": round(1 - r.distance, 4),
            "content": r.content,
        }
        for r in rows
    ]


def execute_tool_call(name: str, arguments: dict) -> str:
    if name == "search":
        results = do_search(arguments["query"])
    elif name == "search_in_book":
        results = do_search_in_book(arguments["book_name"], arguments["query"])
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return json.dumps(results, ensure_ascii=False)


def build_system_prompt(documents: list[dict]) -> str:
    doc_list = "\n".join(
        f"  - {d['filename']} ({d['page_count']} pages)" for d in documents
    )
    return (
        "You are a helpful assistant with access to a document library. "
        "You can search through the documents to answer user questions.\n\n"
        "Available documents:\n"
        f"{doc_list}\n\n"
        "Use the `search` tool to find relevant information across all documents, "
        "or `search_in_book` to search within a specific document by its filename. "
        "Always cite which document and page your information comes from. "
        "If you can't find relevant information, say so honestly."
    )


def main():
    documents = get_document_list()
    if not documents:
        print("No documents found in the database. Run 'make ingest' first.")
        sys.exit(1)

    system_prompt = build_system_prompt(documents)

    print("PDF Search Chatbot")
    print("=" * 40)
    print(f"Loaded {len(documents)} document(s):")
    for d in documents:
        print(f"  - {d['filename']} ({d['page_count']} pages)")
    print("=" * 40)
    print("Type your questions (Ctrl+C to exit)\n")

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input(f"{CYAN}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            response = client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,  # type: ignore[arg-type]
                tools=TOOLS,  # type: ignore[arg-type]
            )

            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                messages.append(msg.model_dump())
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name  # type: ignore[attr-defined]
                    fn_args = json.loads(tool_call.function.arguments)  # type: ignore[attr-defined]

                    print(
                        f"  [tool] {fn_name}({json.dumps(fn_args, ensure_ascii=False)})"
                    )
                    result = execute_tool_call(fn_name, fn_args)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                continue

            assistant_text = msg.content or ""
            messages.append({"role": "assistant", "content": assistant_text})
            print(f"\n{GREEN}Assistant:{RESET} {assistant_text}\n")
            break


if __name__ == "__main__":
    main()
