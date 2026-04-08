from __future__ import annotations

from pgvector.psycopg import Vector

from rag_system.core.embeddings import embed_query
from rag_system.models import RetrievedChunk
from rag_system.db.pg import connect


def retrieve_chunks(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    qemb = embed_query(question)
    qv = Vector(qemb)
    out: list[RetrievedChunk] = []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id::text, doc_id, source_path, chunk_index, content,
                       (embedding <=> %s) AS dist
                FROM document_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (qv, qv, top_k),
            )
            for row in cur.fetchall():
                cid, did, spath, cidx, content, dist = row
                # cosine distance d = 1 - cos_sim  =>  score ~ 1 - d
                score = 1.0 - float(dist) if dist is not None else None
                out.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        doc_id=did,
                        source_path=spath,
                        chunk_index=int(cidx),
                        content=content,
                        score=score,
                    )
                )
    return out
