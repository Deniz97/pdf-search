"""Cache layer for pdf2image and OCR results, keyed by document DB id."""

import json
from pathlib import Path

from PIL import Image

from app.config import settings


def _cache_root(doc_id: str) -> Path:
    return Path(settings.cache_dir) / doc_id


def _images_dir(doc_id: str) -> Path:
    return _cache_root(doc_id) / "images"


def _ocr_dir(doc_id: str) -> Path:
    return _cache_root(doc_id) / "ocr"


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
    return (_ocr_dir(doc_id) / "pages.json").exists()


def save_ocr(doc_id: str, pages: list[tuple[int, str]]) -> None:
    d = _ocr_dir(doc_id)
    d.mkdir(parents=True, exist_ok=True)
    data = [{"page_number": pn, "text": txt} for pn, txt in pages]
    (d / "pages.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_ocr(doc_id: str) -> list[tuple[int, str]]:
    data = json.loads((_ocr_dir(doc_id) / "pages.json").read_text())
    return [(item["page_number"], item["text"]) for item in data]
