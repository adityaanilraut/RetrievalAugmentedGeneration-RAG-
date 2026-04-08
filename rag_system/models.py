from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="Directory or file path to ingest")


class IngestResponse(BaseModel):
    ok: bool = True
    files_processed: int = 0
    chunks_written: int = 0
    message: str = ""


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    chunk_index: int
    content: str
    score: float | None = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    raw_citation_ids: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    database: str = "unknown"


class ErrorResponse(BaseModel):
    detail: str


# CLI / eval (BEIR dense retrieval)
class EvalBeirResult(BaseModel):
    dataset: str = "scifact"
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)
    map_at_k: dict[str, float] = Field(default_factory=dict)
    precision_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float = 0.0
    num_queries: int = 0
    notes: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


# Backward-compatible alias (SciFact runs set dataset to scifact)
EvalSciFactResult = EvalBeirResult
