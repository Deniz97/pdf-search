"""PDF processing: pdf2image -> OCR (Tesseract or PP-StructureV3) -> chunks.

Default: Tesseract (fast, sentence-window chunks).
Optional: PP-StructureV3 (layout+table blocks, slower, needs pip install .[ppstructure]).
"""

import re
import tempfile
from pathlib import Path

import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

from app.config import settings
from app.utils import cache

# PPStructureV3 block types we embed (exclude figure/image)
SEARCHABLE_TYPES = frozenset({"text", "title", "table"})
LABEL_TO_TYPE = {
    "text": "text",
    "paragraph_title": "title",
    "doc_title": "title",
    "title": "title",
    "table": "table",
}


# ---- Tesseract pipeline (default, fast) ----


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _ocr_images_tesseract(
    images: list[Image.Image], doc_id: str | None
) -> list[tuple[int, str]]:
    """Run Tesseract OCR on page images."""
    if doc_id and cache.has_cached_ocr(doc_id):
        print(f"  Using cached OCR for {doc_id}")
        return cache.load_ocr(doc_id)

    pages = []
    for i, image in tqdm(
        enumerate(images, start=1), total=len(images), desc="  OCR pages", unit="page"
    ):
        text = pytesseract.image_to_string(image)
        if text.strip():
            pages.append((i, text.strip()))

    if doc_id:
        cache.save_ocr(doc_id, pages)
    return pages


def _tesseract_pages_to_chunks(
    pages: list[tuple[int, str]],
) -> list[dict]:
    """Chunk Tesseract page text: size=max(50 words, 5 sentences), overlap=max(10 words, 1 sentence)."""
    all_sentences: list[tuple[int, str]] = []
    for page_num, text in pages:
        for sentence in _split_into_sentences(text):
            all_sentences.append((page_num, sentence))

    if not all_sentences:
        return []

    min_chunk_words = settings.chunk_min_words
    min_chunk_sentences = settings.chunk_min_sentences
    min_overlap_words = settings.overlap_min_words
    min_overlap_sentences = settings.overlap_min_sentences

    chunks = []
    chunk_index = 0
    idx = 0

    while idx < len(all_sentences):
        # Build chunk: need both >= min_chunk_words AND >= min_chunk_sentences
        word_count = 0
        end = idx
        while end < len(all_sentences):
            s = all_sentences[end][1]
            word_count += len(s.split())
            end += 1
            if (end - idx) >= min_chunk_sentences and word_count >= min_chunk_words:
                break

        if end <= idx:
            break

        window = all_sentences[idx:end]
        content = " ".join(s for _, s in window)
        page_number = window[0][0]
        chunks.append(
            {
                "content": content,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_type": "text",
                "bbox": None,
                "order_index": chunk_index,
            }
        )
        chunk_index += 1

        # Overlap: step back by max(10 words, 1 sentence) from end of chunk
        overlap_words = 0
        overlap_sentences = 0
        step_back = 0
        for i in range(end - 1, idx - 1, -1):
            step_back += 1
            overlap_words += len(all_sentences[i][1].split())
            overlap_sentences += 1
            if (
                overlap_sentences >= min_overlap_sentences
                and overlap_words >= min_overlap_words
            ):
                break

        next_idx = end - step_back
        # Ensure we always advance (avoids infinite loop when tail has < min_chunk_sentences)
        idx = next_idx if next_idx > idx else end
        if idx >= len(all_sentences):
            break
    return chunks


# ---- PP-StructureV3 pipeline (optional, layout+table) ----


def _get_ppstructure():
    """Lazy singleton for PP-StructureV3 (CPU-only). Requires pip install .[ppstructure]."""
    if not hasattr(_get_ppstructure, "_engine"):
        try:
            from paddleocr import PPStructureV3
        except ImportError as e:
            raise ImportError(
                "PP-Structure backend requires: pip install '.[ppstructure]' or uv add pdf-search-api[ppstructure]"
            ) from e
        _get_ppstructure._engine = PPStructureV3(lang="en", device="cpu")
    return _get_ppstructure._engine


def _point_in_box(x: float, y: float, box: list) -> bool:
    if not box or len(box) < 4:
        return False
    if isinstance(box[0], (list, tuple)):
        xs, ys = [p[0] for p in box[:4]], [p[1] for p in box[:4]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    else:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_center(box) -> tuple[float, float]:
    if box is None:
        return 0.0, 0.0
    if hasattr(box, "tolist"):
        box = box.tolist()
    if isinstance(box, (list, tuple)) and box:
        p0 = box[0]
        if hasattr(p0, "tolist"):
            p0 = p0.tolist()
        if isinstance(p0, (list, tuple)):
            xs = [
                float(p[0]) if not hasattr(p, "tolist") else float(p.tolist()[0])
                for p in box[:4]
            ]
            ys = [
                float(p[1]) if not hasattr(p, "tolist") else float(p.tolist()[1])
                for p in box[:4]
            ]
            return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if len(box) >= 4:
            return (float(box[0]) + float(box[2])) / 2, (
                float(box[1]) + float(box[3])
            ) / 2
    return 0.0, 0.0


def _html_table_to_text(html: str) -> str:
    text = re.sub(r"</tr>\s*", " ", html or "", flags=re.I)
    text = re.sub(r"</t[dh]>\s*", " ", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split()).strip()


def _parse_ppstructurev3_result(
    result, rec_texts: list | None, rec_boxes
) -> list[dict]:
    blocks = []
    res = getattr(result, "res", None) or result
    layout_res = (
        res.get("layout_det_res")
        if hasattr(res, "get")
        else getattr(res, "layout_det_res", None)
    ) or {}
    layout_boxes = (
        layout_res.get("boxes")
        if hasattr(layout_res, "get")
        else getattr(layout_res, "boxes", None)
    ) or []

    if rec_texts is None or rec_boxes is None:
        ocr_res = (
            res.get("overall_ocr_res")
            if hasattr(res, "get")
            else getattr(res, "overall_ocr_res", None)
        ) or {}
        rec_texts = (
            ocr_res.get("rec_texts")
            if hasattr(ocr_res, "get")
            else getattr(ocr_res, "rec_texts", None)
        ) or []
        rec_boxes = (
            ocr_res.get("rec_boxes") or ocr_res.get("rec_polys")
            if hasattr(ocr_res, "get")
            else getattr(ocr_res, "rec_boxes", None)
            or getattr(ocr_res, "rec_polys", None)
        ) or []
        if hasattr(rec_boxes, "tolist"):
            rec_boxes = rec_boxes.tolist()
        elif hasattr(rec_boxes, "__iter__") and not isinstance(
            rec_boxes, (list, tuple)
        ):
            rec_boxes = list(rec_boxes)

    for lb in layout_boxes:
        label = (lb.get("label") or "text").lower()
        chunk_type = LABEL_TO_TYPE.get(label)
        if chunk_type not in SEARCHABLE_TYPES:
            continue
        coord = lb.get("coordinate") or lb.get("bbox")
        if isinstance(coord, (list, tuple)) and len(coord) >= 4:
            bbox = (
                [
                    min(p[0] for p in coord[:4]),
                    min(p[1] for p in coord[:4]),
                    max(p[0] for p in coord[:4]),
                    max(p[1] for p in coord[:4]),
                ]
                if coord and isinstance(coord[0], (list, tuple))
                else list(coord[:4])
            )
        else:
            bbox = None
        texts = []
        for i, txt in enumerate(rec_texts):
            if i < len(rec_boxes):
                rb = rec_boxes[i]
                cx, cy = _box_center(rb)
                if _point_in_box(cx, cy, coord or []):
                    if txt and str(txt).strip():
                        texts.append(str(txt).strip())
        content = " ".join(texts).strip()
        if chunk_type == "table" and not content:
            tbl = lb.get("res") or {}
            content = (
                _html_table_to_text(tbl.get("html") or tbl.get("text") or "")
                if isinstance(tbl, dict)
                else _html_table_to_text(str(tbl))
            )
        if content:
            blocks.append({"type": chunk_type, "content": content, "bbox": bbox})
    return blocks


def _run_ppstructure_on_images(
    images: list[Image.Image], doc_id: str | None
) -> list[tuple[int, list[dict]]]:
    if doc_id and cache.has_cached_ppstructure(doc_id):
        print(f"  Using cached PP-Structure for {doc_id}")
        return cache.load_ppstructure(doc_id)

    engine = _get_ppstructure()
    pages_blocks: list[tuple[int, list[dict]]] = []
    for i, image in tqdm(
        enumerate(images, start=1),
        total=len(images),
        desc="  PP-Structure pages",
        unit="page",
    ):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            path = f.name
        try:
            results = engine.predict(path)
            page_blocks = []
            for res in results or []:
                page_blocks.extend(_parse_ppstructurev3_result(res, None, None))
            pages_blocks.append((i, page_blocks))
        finally:
            Path(path).unlink(missing_ok=True)
    if doc_id:
        cache.save_ppstructure(doc_id, pages_blocks)
    return pages_blocks


def _ppstructure_blocks_to_chunks(
    pages_blocks: list[tuple[int, list[dict]]],
) -> list[dict]:
    chunks = []
    chunk_index = 0
    for page_num, blocks in pages_blocks:
        for order, block in enumerate(blocks):
            content = (block.get("content") or "").strip()
            if not content:
                continue
            chunks.append(
                {
                    "content": content,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "chunk_type": block.get("type") or "text",
                    "bbox": block.get("bbox"),
                    "order_index": order,
                }
            )
            chunk_index += 1
    return chunks


# ---- Public API ----


def pdf_to_images(pdf_path: str, doc_id: str | None = None) -> list[Image.Image]:
    """Convert PDF to page images (cached when doc_id provided)."""
    dpi = getattr(settings, "pdf_dpi", 200) or 200
    if doc_id and cache.has_cached_images(doc_id):
        print(f"  Using cached images for {doc_id}")
        return cache.load_images(doc_id)
    images = convert_from_path(pdf_path, dpi=dpi)
    if doc_id:
        cache.save_images(doc_id, images)
    return images


def extract_chunks_from_pdf(
    pdf_path: str,
    doc_id: str | None = None,
) -> list[dict]:
    """Extract chunks from PDF. Uses Tesseract (default) or PP-Structure per ocr_backend."""
    backend = (getattr(settings, "ocr_backend", "tesseract") or "tesseract").lower()

    # Tesseract with cached OCR: skip loading images entirely (saves memory for large PDFs)
    if backend == "tesseract" and doc_id and cache.has_cached_ocr(doc_id):
        print(f"  Using cached OCR for {doc_id}")
        pages = cache.load_ocr(doc_id)
        print(f"  Loaded OCR: {len(pages)} pages")
        chunks = _tesseract_pages_to_chunks(pages)
        print(f"  Built {len(chunks)} chunks")
        return chunks

    images = pdf_to_images(pdf_path, doc_id)
    if not images:
        return []

    if backend == "ppstructure":
        pages_blocks = _run_ppstructure_on_images(images, doc_id)
        return _ppstructure_blocks_to_chunks(pages_blocks)
    # default: tesseract
    pages = _ocr_images_tesseract(images, doc_id)
    return _tesseract_pages_to_chunks(pages)
