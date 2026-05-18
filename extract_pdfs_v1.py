"""
Crisis Response RAG — Smart PDF Extraction Script v3
=====================================================
Uses pymupdf4llm for clean structured PDFs (TCCC, Sphere, IFRC)
Uses pdftotext -layout for complex column layouts (WASH, Survival Manual)

Install dependencies:
    pip install pymupdf4llm pdfplumber
    poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases
    Add poppler bin/ folder to your system PATH

Output: /extracted_texts/ — one .md or .txt per PDF, clean and RAG-ready
"""

import os
import re
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — update paths to match your machine
# ─────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Aamna\Documents\Resilience Engine\data"
OUTPUT_DIR = r"C:\Users\Aamna\Documents\Resilience Engine\extracted_texts"


WASH_INCLUDE_FILES = ["who-tn-05-emergency-treatment-of-drinking-water-at-the-point-of-use.pdf", "who-tn-07-solid-waste-management-in-emergencies.pdf", "who-tn-08-disposal-of-dead-bodies.pdf", "who-tn-09-how-much-water-is-needed.pdf", "who-tn-13-planning-for-excreta-disposal-in-emergencies.pdf"]

# Pages with fewer words than this are skipped
# (chapter separator photos, image-only pages)
MIN_WORDS_PER_PAGE = 40

# ─────────────────────────────────────────────


# ── Utilities ─────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def clean_text(text: str) -> str:
    """Remove TOC artifacts and excessive whitespace."""
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove TOC dot/dash lines like "Chapter 4 ........... 12"
    text = re.sub(r'^[\.\-\s]{5,}$', '', text, flags=re.MULTILINE)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def tag_special_boxes(text: str) -> str:
    """
    Tag CAUTION/WARNING/TIP/NOTE boxes so Gemma treats them as high-priority.
    Critical for survival/medical content.
    """
    for tag in ["CAUTION", "WARNING", "TIP", "NOTE", "IMPORTANT"]:
        text = re.sub(
            rf'(?m)^({tag}[:\s].+?)(\n\n)',
            rf'[{tag}] \1\2',
            text
        )
    return text


# ── pymupdf4llm extraction (TCCC, Sphere, IFRC) ──

def extract_pymupdf4llm(pdf_path: Path, out_path: Path, source_name: str):
    """
    Best for clean structured PDFs — outputs Markdown.
    Headers, tables, bold text all preserved as proper Markdown.
    Each page comes with metadata (page number, source).
    """
    try:
        import pymupdf4llm
    except ImportError:
        print("pymupdf4llm not installed. Run: pip install pymupdf4llm")
        return

    print(f"  [pymupdf4llm] {pdf_path.name}")

    # page_chunks=True gives us one dict per page with text + metadata
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    kept = 0
    skipped = 0
    output_parts = []

    for page in pages:
        text = page.get("text", "")

        # Skip chapter separator / image-only pages
        if word_count(text) < MIN_WORDS_PER_PAGE:
            skipped += 1
            continue

        cleaned = clean_text(text)

        if source_name == "Survival Manual":
            cleaned = tag_special_boxes(cleaned)

        page_num = page.get("metadata", {}).get("page", "?")
        output_parts.append(f"<!-- page {page_num} -->\n{cleaned}\n")
        kept += 1

    # Save as .md — markdown format works better for chunking later
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(output_parts), encoding="utf-8")

    print(f"{md_path.name} — kept {kept} pages, skipped {skipped} (separators/images)")


# ── pdftotext extraction (WASH, Survival Manual) ──

def extract_pdftotext_layout(pdf_path: Path, out_path: Path, source_name: str):
    """
    Best for 3-column layouts (WASH) and illustrated manuals (Survival Manual).
    pdftotext -layout preserves spatial column order correctly.
    """
    print(f"  [pdftotext -layout] {pdf_path.name}")

    tmp_path = out_path.with_suffix(".tmp.txt")

    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), str(tmp_path)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"pdftotext failed: {result.stderr.strip()}")
        print(f"  → Is poppler installed and in PATH? Check: pdftotext --version")
        tmp_path.unlink(missing_ok=True)
        return

    raw = tmp_path.read_text(encoding="utf-8", errors="ignore")
    tmp_path.unlink(missing_ok=True)

    # pdftotext uses \f as page separator
    pages = raw.split("\f")
    kept_pages = []

    for i, page_text in enumerate(pages):
        if word_count(page_text) < MIN_WORDS_PER_PAGE:
            continue

        cleaned = clean_text(page_text)

        if source_name == "Survival Manual":
            cleaned = tag_special_boxes(cleaned)

        kept_pages.append(f"[PAGE {i+1}]\n{cleaned}\n")

    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text("\n".join(kept_pages), encoding="utf-8")
    print(f"{txt_path.name} — kept {len(kept_pages)} pages")


# ── Strategy map ──────────────────────────────

STRATEGY = {
    "TCCC":            extract_pymupdf4llm,       # clean bullets → markdown
    "Sphere":          extract_pymupdf4llm,        # tables + text → markdown
    "IRFC":            extract_pymupdf4llm,         # tables + text → markdown
    "Survival Manual": extract_pdftotext_layout,   # images + columns → txt
    "WASH":            extract_pdftotext_layout,   # 3-column → txt
}


def should_include(folder_name: str, filename: str) -> bool:
    """Only extract specified WASH chapters."""
    if folder_name == "WASH" and WASH_INCLUDE_FILES:
        return filename in WASH_INCLUDE_FILES
    return True


# ── Main ──────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_path = Path(DATA_DIR)
    out_path  = Path(OUTPUT_DIR)

    total_extracted = 0
    total_skipped   = 0

    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir() or folder.name == "assets":
            continue

        folder_name = folder.name
        extractor   = STRATEGY.get(folder_name)

        if not extractor:
            print(f"\nNo strategy for '{folder_name}' — skipping")
            continue

        print(f"\n{folder_name}  →  {'pymupdf4llm (.md)' if extractor == extract_pymupdf4llm else 'pdftotext (.txt)'}")

        for pdf_file in sorted(folder.glob("*.pdf")):
            if not should_include(folder_name, pdf_file.name):
                print(f"Skipping {pdf_file.name}")
                total_skipped += 1
                continue

            # Check if already extracted (either .md or .txt)
            out_stem     = out_path / f"{folder_name}__{pdf_file.stem}"
            already_done = out_stem.with_suffix(".md").exists() or \
                           out_stem.with_suffix(".txt").exists()

            if already_done:
                print(f"Already done: {out_stem.name}")
                continue

            extractor(pdf_file, out_stem, folder_name)
            total_extracted += 1

    print(f"\n{'─'*50}")
    print(f"   Extraction complete!")
    print(f"   PDFs extracted : {total_extracted}")
    print(f"   PDFs skipped   : {total_skipped}")
    print(f"   Output folder  : {OUTPUT_DIR}")
    print(f"\nNext step: run chunk_and_embed.py to build your ChromaDB")


if __name__ == "__main__":
    # Dependency check
    missing = []
    try:
        import pymupdf4llm
    except ImportError:
        missing.append("pymupdf4llm")

    if missing:
        print(f"Missing packages: pip install {' '.join(missing)}")
        exit(1)

    run()
