"""Document parsing and ingestion into the vector store."""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Tuple

from .store import get_store


def _load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _file_hash(filepath: Path) -> str:
    """MD5 of file content for dedup."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, trying to break at sentence boundaries."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    chunks: List[str] = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at sentence end within the last 20% of the chunk
        chunk_slice = text[start:end]
        search_start = int(len(chunk_slice) * 0.8)
        break_chars = "。！？\n.!?" if any("一" <= c <= "鿿" for c in chunk_slice) else ".!?\n"
        best = -1
        for ch in break_chars:
            pos = chunk_slice.rfind(ch, search_start)
            if pos > best:
                best = pos
        if best > 0:
            end = start + best + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def _parse_pdf(filepath: Path) -> str:
    """Extract text from PDF using pdfplumber (text layer) with pypdf + OCR fallback."""
    texts: list[str] = []

    # 1. Try pdfplumber (best for text-based PDFs)
    try:
        import logging
        logging.getLogger("pdfplumber").setLevel(logging.ERROR)
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
        if texts:
            return "\n\n".join(texts)
    except Exception:
        pass

    # 2. Fallback to pypdf
    from pypdf import PdfReader
    reader = PdfReader(str(filepath))
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t)
    if texts:
        return "\n\n".join(texts)

    # 3. No text layer — try OCR on scanned/image PDF
    ocr_text = _pdf_ocr_fallback(filepath)
    if ocr_text:
        return ocr_text

    return ""


def _pdf_ocr_fallback(filepath: Path) -> str:
    """Try OCR via Tesseract for scanned PDFs. Returns empty string if unavailable."""
    try:
        import subprocess
        import tempfile
        import shutil

        # Check if tesseract is installed
        if shutil.which("tesseract") is None:
            print("  [hint] PDF has no text layer. Install Tesseract for OCR support:")
            print("         https://github.com/UB-Mannheim/tesseract/wiki")
            print("         Then: pip install pytesseract pdf2image")
            return ""

        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(filepath, dpi=200, first_page=1, last_page=50)
        texts = []
        for img in images:
            t = pytesseract.image_to_string(img, lang="chi_sim+eng")
            if t.strip():
                texts.append(t.strip())
        return "\n\n".join(texts) if texts else ""
    except ImportError:
        print("  [hint] PDF has no text layer. pip install pytesseract pdf2image")
        print("         + install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki")
        return ""
    except Exception as e:
        print(f"  [ocr error] {e}")
        return ""


def _parse_markdown(filepath: Path) -> str:
    """Read markdown as plain text (structure preserved as-is)."""
    return filepath.read_text(encoding="utf-8", errors="replace")


def _parse_txt(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="replace")


PARSERS = {
    ".pdf": _parse_pdf,
    ".md": _parse_markdown,
    ".txt": _parse_txt,
    ".py": _parse_txt,
    ".js": _parse_txt,
    ".ts": _parse_txt,
    ".go": _parse_txt,
    ".rs": _parse_txt,
    ".java": _parse_txt,
    ".cpp": _parse_txt,
    ".c": _parse_txt,
    ".h": _parse_txt,
    ".json": _parse_txt,
    ".yaml": _parse_txt,
    ".yml": _parse_txt,
    ".toml": _parse_txt,
    ".xml": _parse_txt,
    ".html": _parse_txt,
    ".css": _parse_txt,
    ".sql": _parse_txt,
    ".sh": _parse_txt,
}


def ingest_file(filepath: Path) -> int:
    """Ingest a single file. Returns number of chunks added."""
    suffix = filepath.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        print(f"  [skip] unsupported format: {suffix}")
        return 0

    try:
        text = parser(filepath)
    except Exception as e:
        print(f"  [error] parsing {filepath.name}: {e}")
        return 0

    if not text or not text.strip():
        print(f"  [skip] no text extracted from {filepath.name}")
        return 0

    cfg = _load_config()["rag"]
    chunks = _chunk_text(text, cfg["chunk_size"], cfg["chunk_overlap"])
    if not chunks:
        return 0

    collection, _ = get_store()
    fhash = _file_hash(filepath)

    # Remove old chunks for this file if re-ingesting
    try:
        results = collection.get(where={"source_hash": fhash})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception:
        pass

    ids = [f"{fhash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": str(filepath),
            "source_name": filepath.name,
            "source_hash": fhash,
            "chunk_index": i,
            "chunk_count": len(chunks),
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def ingest_directory(dirpath: Path, recursive: bool = True) -> Tuple[int, int]:
    """Ingest all supported files in a directory. Returns (files_processed, total_chunks)."""
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in dirpath.glob(pattern)
        if p.is_file() and p.suffix.lower() in PARSERS
    ]
    total_files = 0
    total_chunks = 0
    for fp in sorted(files):
        print(f"  ingesting {fp.name} ...")
        n = ingest_file(fp)
        if n > 0:
            total_files += 1
            total_chunks += n
            print(f"    -> {n} chunks")
    return total_files, total_chunks
