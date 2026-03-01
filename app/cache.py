"""Cache layer for pdf2image and OCR results, keyed by document DB id."""

from pathlib import Path

from PIL import Image

from app.config import settings


def _cache_root(doc_id: str) -> Path:
    return Path(settings.cache_dir) / doc_id


def _images_dir(doc_id: str) -> Path:
    return _cache_root(doc_id) / "images"


def _text_dir(doc_id: str) -> Path:
    return _cache_root(doc_id) / "text"


# -- pdf2image cache --


def has_cached_images(doc_id: str) -> bool:
    d = _images_dir(doc_id)
    return d.exists() and any(d.glob("*.png"))


def save_images(doc_id: str, images: list[Image.Image]) -> None:
    d = _images_dir(doc_id)
    d.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images, start=1):
        img.save(d / f"page_{i:04d}.png", "PNG")


def load_images(doc_id: str) -> list[Image.Image]:
    d = _images_dir(doc_id)
    paths = sorted(d.glob("page_*.png"))
    return [Image.open(p) for p in paths]


# -- OCR text cache --


def has_cached_ocr(doc_id: str) -> bool:
    """Check if OCR text files exist for this document."""
    text_dir = _text_dir(doc_id)
    return text_dir.exists() and any(text_dir.glob("page_*.txt"))


def save_ocr(doc_id: str, pages: list[tuple[int, str]]) -> None:
    """Save OCR results as individual text files, one per page."""
    text_dir = _text_dir(doc_id)
    text_dir.mkdir(parents=True, exist_ok=True)
    for page_number, text in pages:
        text_file = text_dir / f"page_{page_number:04d}.txt"
        text_file.write_text(text, encoding="utf-8")


def load_ocr(doc_id: str) -> list[tuple[int, str]]:
    """Load OCR results from individual text files."""
    text_dir = _text_dir(doc_id)
    pages = []
    for text_file in sorted(text_dir.glob("page_*.txt")):
        # Extract page number from filename (e.g., "page_0001.txt" -> 1)
        page_number = int(text_file.stem.split("_")[1])
        text = text_file.read_text(encoding="utf-8")
        pages.append((page_number, text))
    return pages
