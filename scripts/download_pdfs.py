#!/usr/bin/env python3
"""Download PDFs from Google search results (PDF-only, under 20 pages).

Uses Selenium to open Chrome, run Google searches, and fetch result URLs.
Downloads up to 3 PDFs per run that have fewer than 20 pages.

Tracks downloads in the `downloaded_pdfs` table (url, search_query, local_path)
to skip already-downloaded links on subsequent runs.

Requires: make migrate (creates downloaded_pdfs table), database running.
Requires: Chrome/Chromium installed; chromedriver (or Selenium 4.6+ auto-managed).

Usage:
    uv run python scripts/download_pdfs.py
    # or: make download-pdfs
"""
import sys
import time
from pathlib import Path

# Add project root so `app` is importable when run as scripts/download_pdfs.py
_srcdir = Path(__file__).resolve().parent.parent
if str(_srcdir) not in sys.path:
    sys.path.insert(0, str(_srcdir))

import re
import tempfile
import random
import traceback
from urllib.parse import parse_qs, urlparse

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tqdm import tqdm

from app.config import settings
from pypdf import PdfReader

# ---- Selenium selectors (adjust if Google's DOM changes) ----
GOOGLE_URL = "https://www.google.com"
SEARCH_INPUT_SELECTORS = [
    (By.CSS_SELECTOR, "textarea[name='q']"),
    (By.CSS_SELECTOR, "input[name='q']"),
]  # try in order
CONSENT_BUTTON_SELECTORS = [
    (By.CSS_SELECTOR, "button#L2AGLb"),  # "Accept all" (common)
    (By.CSS_SELECTOR, "button[id='W0wltc']"),  # Reject/other variant
]  # click first found to dismiss cookie dialog
RESULT_CONTAINER_SELECTOR = (By.CSS_SELECTOR, "div.kCrYT")  # each result block (div.egMi0.kCrYT)
RESULT_LINK_SELECTOR = (By.CSS_SELECTOR, "a[href*='url?q=']")  # Google redirect link
RESULT_TITLE_SELECTOR = (By.CSS_SELECTOR, "h3")  # title within each result
SELENIUM_WAIT_SECONDS = 15
SELENIUM_HEADLESS = False  # set False to watch the browser

# ---- Configure these ----
COUNTRIES = [
    "United States",
    "China",
    "Russia",
    "Iran",
    "Syria",
    "Turkey",
    "Israel",
    "Saudi Arabia",
    "United Arab Emirates",
    "Qatar",
]

EVENTS = [
    "Gulf War",
    "Israeli-Palestinian Conflict",
    "Syrian Civil War",
    "Petrodollar System",
    "OPEC",
    "BRICS",
    "Defeat of Asad Regime",
    "Kurdish Independence Movement",
    "Arab Spring",
    "Iran Israel Conflict",
    "Abraham Accords",
    # Contemporary Israel / Palestine
    "October 7 Attacks",
    "Gaza War 2023",
    "Hamas Israel Conflict",
    "West Bank Settlements",
    "Two-State Solution",
    "Jerusalem Status",
    "Al-Aqsa Tensions",
    "Gaza Blockade",
    "Palestinian Statehood",
    # Iran
    "Iran Nuclear Deal JCPOA",
    "Iran Nuclear Program",
    "Iran Uranium Enrichment",
    "Iran Sanctions",
    "Iran Proxy Network",
    "Axis of Resistance",
    "Iran Israel Tensions",
    "Iran Drone Strikes",
    "Iran China Agreement",
    # Saudi Arabia
    "Saudi Vision 2030",
    "Saudi Aramco Attack 2019",
    "Saudi Oil Policy",
    "Saudi Oil Production Cuts",
    "Saudi Iran Rapprochement",
    "Beijing Saudi Iran Agreement",
    # Yemen / Houthi
    "Yemen Civil War",
    "Houthi Rebellion",
    "Red Sea Houthi Attacks",
    "Houthi Maritime Attacks",
    # Gulf / Regional
    "Qatar Crisis",
    "Gulf Rift",
    "UAE Israel Normalization",
    "Bahrain Israel Normalization",
    "Hezbollah Israel Conflict",
    "Lebanon Economic Crisis",
    "Sunni Shia Divide",
    "Regional Power Rivalry",
    "US Troop Withdrawal Middle East",
]
ASPECTS = ["Geopolitics", "Energy", "Finance", "Military", "Intelligence"]

SEARCH_QUERIES = [f"{c} on {e} by {a}" for c in COUNTRIES for e in EVENTS for a in ASPECTS]
random.shuffle(SEARCH_QUERIES)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "downloaded-pdfs"
MAX_PAGES = 20
TARGET_COUNT = 3
NUM_SEARCH_RESULTS = 20  # How many URLs to fetch per query
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def create_driver() -> webdriver.Chrome:
    """Create and return a Chrome WebDriver. Caller must quit() when done."""
    options = Options()
    if SELENIUM_HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"user-agent={REQUEST_HEADERS['User-Agent']}")

    return webdriver.Chrome(options=options)


def resolve_google_redirect_url(href: str | None) -> str | None:
    """Extract actual URL from Google's /url?q=... redirect. Return None if not a redirect."""
    if not href or "url?q=" not in href:
        return None
    try:
        parsed = urlparse(href)
        params = parse_qs(parsed.query)
        urls = params.get("q", [])
        if urls:
            return urls[0]
    except Exception:
        pass
    return None


def dismiss_consent_if_present(driver: webdriver.Chrome) -> None:
    """Click consent/cookie dialog if present (e.g. EU/UK)."""
    for by, selector in CONSENT_BUTTON_SELECTORS:
        try:
            btn = driver.find_element(by, selector)
            if btn.is_displayed():
                btn.click()
                time.sleep(1)
                return
        except Exception:
            continue


def search_with_selenium(driver: webdriver.Chrome, query: str, num_results: int = 20) -> list[dict]:
    """Use existing Chrome driver to search Google, return list of {url, title} dicts."""
    driver.get(GOOGLE_URL)
    time.sleep(1)
    wait = WebDriverWait(driver, SELENIUM_WAIT_SECONDS)

    dismiss_consent_if_present(driver)

    # Find search input (try multiple selectors)
    search_box = None
    for by, selector in SEARCH_INPUT_SELECTORS:
        try:
            search_box = wait.until(EC.presence_of_element_located((by, selector)))
            if search_box and search_box.is_displayed():
                break
        except Exception:
            continue
    if not search_box:
        raise RuntimeError("Could not find Google search input (tried textarea and input[name='q'])")

    search_box.clear()
    search_box.send_keys(query)
    time.sleep(0.5)
    search_box.submit()
    time.sleep(2)

    results: list[dict] = []
    containers = driver.find_elements(*RESULT_CONTAINER_SELECTOR)

    for el in containers:
        if len(results) >= num_results:
            break
        try:
            link = el.find_element(*RESULT_LINK_SELECTOR)
            raw_href = link.get_attribute("href")
            # Google uses /url?q=ACTUAL_URL redirect; extract real URL
            href = resolve_google_redirect_url(raw_href)
            if not href:
                # Direct link (no redirect)
                href = raw_href if raw_href and raw_href.startswith("http") and "google.com" not in raw_href else None
            if not href:
                continue

            titles = el.find_elements(*RESULT_TITLE_SELECTOR)
            title = titles[0].text if titles else "(no title)"
            results.append({"url": href, "title": title})
        except Exception:
            continue

    return results


def ensure_filetype_pdf(query: str) -> str:
    """Ensure filetype:pdf is in the query (append if missing)."""
    if "filetype:pdf" not in query.lower():
        return f"{query} filetype:pdf"
    return query


def is_pdf_url(url: str) -> bool:
    """Check if URL likely points to a PDF (by extension or path)."""
    u = url.lower().split("?")[0]
    return u.endswith(".pdf") or ".pdf" in u.rsplit("/", 1)[-1]


def get_page_count(pdf_path: Path) -> int | None:
    """Return page count of PDF, or None if unreadable."""
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return None


def download_pdf(url: str, dest: Path) -> bool:
    """Download PDF from URL to dest. Return True on success."""
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        r.raise_for_status()
        content = r.content
        if not content or len(content) < 100:
            return False
        # Basic PDF magic bytes check
        if not content[:4].startswith(b"%PDF"):
            return False
        dest.write_bytes(content)
        return True
    except Exception:
        return False


def download_pdf_with_progress(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download PDF with tqdm progress bar. Return True on success."""
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        content = []
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    content.append(chunk)
                    pbar.update(len(chunk))
        data = b"".join(content)
        if not data or len(data) < 100:
            return False
        if not data[:4].startswith(b"%PDF"):
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def safe_filename(url: str, index: int) -> str:
    """Generate a safe filename from URL."""
    # Use last path component, sanitize, fallback to index
    name = url.rsplit("/", 1)[-1].split("?")[0]
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    # Remove unsafe chars
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    if not name or name == ".pdf":
        name = f"document_{index}.pdf"
    return name


def get_already_downloaded_urls(session: Session) -> set[str]:
    """Return set of URLs we've already downloaded (from downloaded_pdfs table)."""
    rows = session.execute(text("SELECT url FROM downloaded_pdfs")).fetchall()
    return {r.url for r in rows}


def record_download(session: Session, url: str, search_query: str, local_path: str) -> None:
    """Record a successful download in the database."""
    session.execute(
        text(
            "INSERT INTO downloaded_pdfs (id, url, search_query, local_path) "
            "VALUES (gen_random_uuid(), :url, :search_query, :local_path)"
        ),
        {"url": url, "search_query": search_query, "local_path": local_path},
    )
    session.commit()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        seen_urls = get_already_downloaded_urls(session)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Skipping {len(seen_urls)} already-downloaded URL(s)")
    print(f"Searching for up to {TARGET_COUNT} PDFs under {MAX_PAGES} pages each.\n")

    collected: list[tuple[str, Path]] = []
    driver = create_driver()

    try:
        for query in SEARCH_QUERIES:
            if len(collected) >= TARGET_COUNT:
                break

            search_query = ensure_filetype_pdf(query)

            # ---- 1. Search query ----
            print(f"\n{'='*60}")
            print(f"QUERY: {search_query}")
            print(f"{'='*60}")

            pdf_results: list[dict] = []  # {"url", "title"}

            try:
                raw_results = search_with_selenium(driver, search_query, num_results=NUM_SEARCH_RESULTS)
                raw_count = len(raw_results)
                pdf_count = 0
                dup_count = 0
                for r in raw_results:
                    url = r.get("url", "")
                    title = r.get("title", "")
                    if not url:
                        continue
                    if url in seen_urls:
                        dup_count += 1
                        continue
                    if not is_pdf_url(url):
                        continue
                    pdf_count += 1
                    pdf_results.append({"url": url, "title": title or "(no title)"})
                    seen_urls.add(url)
            except Exception as e:
                print(f"  Search failed: {e}")
                traceback.print_exc()
                continue
            time.sleep(100)
            # ---- 2. Result count ----
            print(f"Raw results: {raw_count} | PDFs: {pdf_count} | Already seen: {dup_count}")
            print(f"Result count: {len(pdf_results)} PDF(s)")

            if not pdf_results:
                continue

            # ---- 3. Top 10 results ----
            top10 = pdf_results[:10]
            print(f"\nTop {len(top10)} results:")
            for i, r in enumerate(top10, 1):
                title_short = (r["title"][:60] + "…") if len(r["title"]) > 60 else r["title"]
                url_short = r["url"][:70] + "…" if len(r["url"]) > 70 else r["url"]
                print(f"  {i:2}. {title_short}")
                print(f"      {url_short}")
                print(f"      pages: -")

            # ---- 4. Selected for downloading ----
            print(f"\nAttempting downloads (target: {TARGET_COUNT}):")
            selected: list[tuple[str, str, Path, int]] = []  # (url, title, dest, pages)

            for idx, r in enumerate(pdf_results, 1):
                if len(collected) >= TARGET_COUNT:
                    break

                url, title = r["url"], r["title"]
                filename = safe_filename(url, len(collected) + 1)
                desc = f"[{idx}/{len(pdf_results)}] {filename[:35]}"

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    if not download_pdf_with_progress(url, tmp_path, desc=desc):
                        continue
                    pages = get_page_count(tmp_path)
                    if pages is None:
                        continue
                    if pages > MAX_PAGES:
                        continue

                    dest = OUTPUT_DIR / filename
                    if dest.exists():
                        stem, suffix = dest.stem, dest.suffix
                        idx = 1
                        while dest.exists():
                            dest = OUTPUT_DIR / f"{stem}_{idx}{suffix}"
                            idx += 1
                    tmp_path.rename(dest)
                    collected.append((url, dest))
                    selected.append((url, title, dest, pages))
                    seen_urls.add(url)
                    with Session(engine) as session:
                        record_download(
                            session,
                            url=url,
                            search_query=query,
                            local_path=str(dest.resolve()),
                        )
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

            # ---- 5 & 6. Selected 3 (or fewer) + finish round ----
            if selected:
                print(f"\nSelected {len(selected)} for this round:")
                for i, (url, title, dest, pages) in enumerate(selected, 1):
                    title_short = (title[:50] + "…") if len(title) > 50 else title
                    print(f"  {i}. {title_short}")
                    print(f"     {url[:75]}…" if len(url) > 75 else f"     {url}")
                    print(f"     → {dest.name} ({pages} pages)")

            print(f"\n--- Round complete (total collected: {len(collected)}/{TARGET_COUNT}) ---")

        print(f"\nDone. Downloaded {len(collected)} PDF(s) to {OUTPUT_DIR}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
