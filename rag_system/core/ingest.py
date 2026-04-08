from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from pgvector.psycopg import Vector

from rag_system.core.chunking import chunk_text
from rag_system.core.embeddings import embed_texts
from rag_system.db.pg import connect


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)


def extract_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _read_pdf(path)
    if suf in (".txt", ".md", ".markdown"):
        return _read_text_file(path)
    raise ValueError(f"Unsupported file type: {path}")


def collect_paths(root: Path) -> list[Path]:
    exts = {".pdf", ".txt", ".md", ".markdown"}
    if root.is_file():
        if root.suffix.lower() in exts:
            return [root]
        raise ValueError(f"Unsupported file: {root}")
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def ingest_path(path_str: str, *, chunk_size: int = 1200, overlap: int = 200) -> tuple[int, int]:
    """Ingest files under path. Returns (files_processed, chunks_written)."""
    root = Path(path_str).expanduser().resolve()
    paths = collect_paths(root)
    files_ok = 0
    chunks_total = 0

    for fp in paths:
        text = extract_text(fp)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            files_ok += 1
            continue
        doc_id = str(fp.resolve())
        embeddings = embed_texts(chunks)

        with connect() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM document_chunks WHERE doc_id = %s", (doc_id,))
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
                    cid = str(uuid4())
                    conn.execute(
                        """
                        INSERT INTO document_chunks
                        (chunk_id, doc_id, source_path, chunk_index, content, embedding)
                        VALUES (%s::uuid, %s, %s, %s, %s, %s)
                        """,
                        (cid, doc_id, str(fp), idx, chunk, Vector(emb)),
                    )
        files_ok += 1
        chunks_total += len(chunks)

    return files_ok, chunks_total
