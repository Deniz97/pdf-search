import re

from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

from app import cache
from app.config import settings


def pdf_to_images(pdf_path: str, doc_id: str) -> list[Image.Image]:
    """Convert PDF to images, using cache if available."""
    if cache.has_cached_images(doc_id):
        print(f"  Using cached images for {doc_id}")
        return cache.load_images(doc_id)

    images = convert_from_path(pdf_path)
    cache.save_images(doc_id, images)
    return images


def ocr_images(images: list[Image.Image], doc_id: str) -> list[tuple[int, str]]:
    """Run OCR on page images, using cache if available."""
    if cache.has_cached_ocr(doc_id):
        print(f"  Using cached OCR for {doc_id}")
        return cache.load_ocr(doc_id)

    pages = []
    for i, image in tqdm(
        enumerate(images, start=1), total=len(images), desc="  OCR pages", unit="page"
    ):
        text = pytesseract.image_to_string(image)
        if text.strip():
            pages.append((i, text.strip()))

    cache.save_ocr(doc_id, pages)
    return pages


def extract_text_from_pdf(
    pdf_path: str, doc_id: str | None = None
) -> list[tuple[int, str]]:
    """Full pipeline: pdf2image -> OCR, with per-step caching when doc_id is provided."""
    if doc_id:
        images = pdf_to_images(pdf_path, doc_id)
        return ocr_images(images, doc_id)

    images = convert_from_path(pdf_path)
    pages = []
    for i, image in tqdm(
        enumerate(images, start=1), total=len(images), desc="  OCR pages", unit="page"
    ):
        text = pytesseract.image_to_string(image)
        if text.strip():
            pages.append((i, text.strip()))
    return pages


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    pages: list[tuple[int, str]],
    chunk_size: int = settings.chunk_sentences,
    overlap: int = settings.chunk_overlap,
) -> list[dict]:
    """Chunk text by sentences with overlap.
    Returns list of dicts with content, page_number, and chunk_index.
    """
    all_sentences: list[tuple[int, str]] = []
    for page_num, text in pages:
        for sentence in split_into_sentences(text):
            all_sentences.append((page_num, sentence))

    if not all_sentences:
        return []

    chunks = []
    idx = 0
    chunk_index = 0

    while idx < len(all_sentences):
        end = min(idx + chunk_size, len(all_sentences))
        window = all_sentences[idx:end]

        content = " ".join(s for _, s in window)
        page_number = window[0][0]

        chunks.append(
            {
                "content": content,
                "page_number": page_number,
                "chunk_index": chunk_index,
            }
        )

        idx += chunk_size - overlap
        chunk_index += 1

    return chunks
