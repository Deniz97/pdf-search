from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from app.database import init_db
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="PDF Search",
    description="Upload PDFs and search through them using natural language",
    version="0.1.0",
    lifespan=lifespan,
)

# Templates and static files
BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# CORS middleware (still useful for API access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page - query box with matched sections."""
    return templates.TemplateResponse(
        request=request,
        name="search.html",
        context={}
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page - interactive chat interface."""
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={}
    )


@app.get("/directories", response_class=HTMLResponse)
async def directories_page(request: Request):
    """Directories management page."""
    return templates.TemplateResponse(
        request=request,
        name="directories.html",
        context={}
    )
