"""
Crisis Response RAG — Chunk & Embed Script
==========================================
Reads extracted .md and .txt files, splits them into
smart topic-based chunks, embeds them with all-MiniLM-L6-v2,
and stores everything in ChromaDB.

Run after extract_pdfs_v3.py has completed.

Usage:
    python chunk_and_embed.py
"""

import os
import re
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EXTRACTED_DIR = r"C:\Users\Aamna\Documents\Resilience Engine\extracted_texts"
CHROMA_DIR    = r"C:\Users\Aamna\Documents\Resilience Engine\chroma_db"

# Embedding model — runs locally, no API key needed
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunk size in characters — tuned for crisis Q&A
# Large enough to include full procedures, small enough to stay focused
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# ChromaDB collection name
COLLECTION_NAME = "crisis_knowledge"

# ─────────────────────────────────────────────


# ── Source metadata map ───────────────────────
# Used to tag each chunk with its source for citations
SOURCE_META = {
    "IRFC":            {"source": "IFRC First Aid Guidelines 2025",     "type": "medical"},
    "Sphere":          {"source": "Sphere Humanitarian Handbook 2018",  "type": "disaster_response"},
    "TCCC":            {"source": "TCCC Guidelines",                    "type": "medical"},
    "Survival Manual": {"source": "US Army Survival Manual FM 21-76",   "type": "survival"},
    "WASH":            {"source": "WHO/WEDC Emergency WASH Guidelines", "type": "water_sanitation"},
}


# ── Chunking strategies ───────────────────────

def chunk_markdown(text: str, filename: str, folder: str) -> list[dict]:
    """
    For .md files (IRFC, Sphere, TCCC).
    First splits by markdown headers to preserve topic boundaries,
    then splits oversized sections by character count.
    """
    # Split by headers first — keeps procedures together
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",  "h1"),
            ("##", "h2"),
            ("###","h3"),
        ],
        strip_headers=False
    )

    # Then split large sections further
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    # First pass: header-based splits
    header_chunks = header_splitter.split_text(text)

    # Second pass: split anything still too large
    final_chunks = []
    for hchunk in header_chunks:
        content = hchunk.page_content
        if len(content) > CHUNK_SIZE:
            sub_chunks = char_splitter.split_text(content)
            for sub in sub_chunks:
                final_chunks.append({
                    "text":     sub,
                    "metadata": hchunk.metadata  # preserves header context
                })
        else:
            final_chunks.append({
                "text":     content,
                "metadata": hchunk.metadata
            })

    return final_chunks


def chunk_text(text: str, filename: str, folder: str) -> list[dict]:
    """
    For .txt files (WASH, Survival Manual).
    Uses recursive character splitter with page markers as boundaries.
    """
    # Remove page markers but use them as split hints
    text = re.sub(r'\[PAGE \d+\]', '\n\n', text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_text(text)
    return [{"text": c, "metadata": {}} for c in chunks]


def clean_chunk(text: str) -> str:
    """Remove noise that hurts embedding quality."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    # Remove page number artifacts like "9.1" or "9.2" on their own line
    text = re.sub(r'^\d+\.\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def is_valid_chunk(text: str) -> bool:
    """Skip chunks that are too short or just noise."""
    words = text.split()
    if len(words) < 15:
        return False
    # Skip chunks that are mostly numbers/special chars (table artifacts)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False
    return True


# ── Main pipeline ─────────────────────────────

def run():
    extracted_path = Path(EXTRACTED_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Load embedding model
    print(f"Loading embedding model: {EMBED_MODEL}")
    print("(First run downloads ~90MB — normal)")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("✅ Model loaded\n")

    # Connect to ChromaDB
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )

    total_chunks = 0
    total_skipped = 0

    # Process each extracted file
    files = sorted(list(extracted_path.glob("*.md")) + list(extracted_path.glob("*.txt")))

    if not files:
        print(f"❌ No files found in {EXTRACTED_DIR}")
        print("   Run extract_pdfs_v3.py first!")
        return

    for filepath in files:
        filename   = filepath.stem                        # e.g. "TCCC__TCCC"
        folder     = filename.split("__")[0]              # e.g. "TCCC"
        source_meta = SOURCE_META.get(folder, {
            "source": filename,
            "type": "unknown"
        })

        print(f"📄 {filename} [{filepath.suffix}]")

        text = filepath.read_text(encoding="utf-8", errors="ignore")

        # Choose chunking strategy based on file type
        if filepath.suffix == ".md":
            raw_chunks = chunk_markdown(text, filename, folder)
        else:
            raw_chunks = chunk_text(text, filename, folder)

        # Filter, clean, and embed
        valid_chunks  = []
        chunk_texts   = []
        chunk_ids     = []
        chunk_metas   = []

        for i, chunk in enumerate(raw_chunks):
            cleaned = clean_chunk(chunk["text"])

            if not is_valid_chunk(cleaned):
                total_skipped += 1
                continue

            chunk_id = f"{filename}__chunk_{i:04d}"

            # Build rich metadata for each chunk
            metadata = {
                "source":   source_meta.get("source", filename),
                "type":     source_meta.get("type", "unknown"),
                "folder":   folder,
                "filename": filename,
                **{k: str(v) for k, v in chunk["metadata"].items()}
            }

            valid_chunks.append(cleaned)
            chunk_texts.append(cleaned)
            chunk_ids.append(chunk_id)
            chunk_metas.append(metadata)

        if not valid_chunks:
            print(f"  ⚠️  No valid chunks found — skipping")
            continue

        # Embed in batches of 64 (memory efficient)
        batch_size = 64
        all_embeddings = []
        for start in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[start:start + batch_size]
            embeddings = embedder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())

        # Store in ChromaDB
        collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=all_embeddings,
            metadatas=chunk_metas,
        )

        kept = len(valid_chunks)
        total_chunks += kept
        print(f"  ✅ {kept} chunks embedded and stored")

    print(f"\n{'─'*50}")
    print(f"✅ Knowledge base built!")
    print(f"   Total chunks stored : {total_chunks}")
    print(f"   Chunks skipped      : {total_skipped}")
    print(f"   ChromaDB location   : {CHROMA_DIR}")
    print(f"   Collection          : {COLLECTION_NAME}")
    print(f"\nNext step: run test_retrieval.py to verify RAG quality")


# ── Quick test query ──────────────────────────

def test_retrieval(query: str = "how to stop bleeding from a wound"):
    """
    Quick sanity check — run after chunk_and_embed completes.
    Call: python chunk_and_embed.py test
    """
    print(f"\n🔍 Test query: '{query}'")

    embedder   = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_vec = embedder.encode([query])[0].tolist()
    results   = collection.query(
        query_embeddings=[query_vec],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n--- Result {i+1} (distance: {dist:.3f}) ---")
        print(f"Source : {meta.get('source', '?')}")
        print(f"Type   : {meta.get('type', '?')}")
        print(f"Text   : {doc[:300]}...")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_retrieval()
    else:
        run()
