from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rag_system.core.generation import generate_answer
from rag_system.core.ingest import ingest_path
from rag_system.core.retrieval import retrieve_chunks
from rag_system.db.pg import check_db, init_db
from rag_system.models import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_db()
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title="RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    db = check_db()
    return HealthResponse(status="ok" if db == "ok" else "degraded", database=db)


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        files, chunks = ingest_path(req.path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e)) from e
    return IngestResponse(
        ok=True,
        files_processed=files,
        chunks_written=chunks,
        message=f"Ingested {files} file(s), {chunks} chunk(s).",
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    top_k = max(1, min(req.top_k, 50))
    chunks = retrieve_chunks(req.question, top_k=top_k)
    return generate_answer(req.question, chunks)
