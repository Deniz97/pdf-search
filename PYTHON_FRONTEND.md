# Python-Native Frontend

The frontend is now fully integrated into the FastAPI application using Jinja2 templates and HTMX. Everything runs in a single Python process - no separate frontend server needed!

## Features

1. **Search Page** (`/`) - Query box that returns PDFs with matched sections
2. **Chat Page** (`/chat`) - Interactive chat interface replicating the CLI chatbot

## Technology Stack

- **FastAPI** - Web framework
- **Jinja2** - Template engine
- **HTMX** - Dynamic interactions without JavaScript framework
- **Static Files** - CSS served directly by FastAPI

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Start the server:**
   ```bash
   make dev
   # or
   uv run uvicorn app.main:app --reload
   ```

3. **Open your browser:**
   - http://localhost:8000 - Search page
   - http://localhost:8000/chat - Chat page
   - http://localhost:8000/api/v1/docs - API documentation

## Project Structure

```
.
├── templates/           # Jinja2 HTML templates
│   ├── base.html       # Base template with navigation
│   ├── search.html     # Search page
│   ├── chat.html       # Chat page
│   └── search_results.html  # Search results fragment
├── static/             # Static files (CSS, JS)
│   └── style.css      # Stylesheet
└── app/
    ├── main.py        # FastAPI app with template routes
    └── routes.py      # API endpoints (also handles HTMX)
```

## How It Works

### Search Page

1. User enters query in form
2. HTMX sends POST request to `/api/v1/documents/search` with JSON body
3. Server detects HTMX request (via `hx-request` header)
4. Returns HTML fragment from `search_results.html` template
5. HTMX swaps the HTML into `#results` div

### Chat Page

1. Uses vanilla JavaScript (no HTMX) for more complex interactions
2. Handles tool calling loop client-side
3. Makes multiple API calls to `/api/v1/chat` and `/api/v1/chat/tool`
4. Updates DOM with messages and tool results

## API Endpoints

All existing API endpoints still work for programmatic access:

- `POST /api/v1/documents/search` - Returns HTML (HTMX) or JSON (API)
- `POST /api/v1/chat` - Chat with tool calling
- `POST /api/v1/chat/tool` - Execute tool calls
- `POST /api/v1/documents/query` - Query with LLM answer
- `GET /api/v1/documents` - List documents
- `POST /api/v1/documents/upload` - Upload PDF

## Advantages

✅ **Single Process** - Everything runs in one Python process  
✅ **No Build Step** - Templates are rendered server-side  
✅ **Simple Deployment** - Just run FastAPI, no separate frontend server  
✅ **Fast Development** - Hot reload works for templates too  
✅ **API Compatible** - Same endpoints work for both web UI and API clients  

## Next Steps (Optional)

- Add file upload UI on search/chat pages
- Add document list page
- Improve error handling and loading states
- Add dark mode toggle
