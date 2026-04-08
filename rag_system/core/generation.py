from __future__ import annotations

import re

from openai import OpenAI

from rag_system.config import get_settings
from rag_system.models import Citation, QueryResponse, RetrievedChunk


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        label = f"S{i}"
        lines.append(f"[{label}] (chunk_id={ch.chunk_id}, file={ch.source_path})\n{ch.content}")
    block = "\n\n---\n\n".join(lines)
    user = f"""Question: {question}

Use ONLY the sources below. If the answer is not in the sources, say you cannot find it in the documents.

Sources:
{block}

Answer concisely. After each sentence or claim that uses a source, add a citation like [S1] or [S2] matching the source label."""
    system = (
        "You are a careful assistant. Ground every factual claim in the provided sources using [S#] citations. "
        "Do not invent facts not present in sources."
    )
    return system, user


def _parse_citation_labels(answer: str) -> set[str]:
    return set(re.findall(r"\[S(\d+)\]", answer))


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> QueryResponse:
    if not chunks:
        return QueryResponse(
            answer="No retrieved context. Ingest documents first.",
            citations=[],
            retrieved=[],
            raw_citation_ids=[],
        )

    system, user = _build_prompt(question, chunks)
    settings = get_settings()
    client = (
        OpenAI(api_key=settings.openai_api_key)
        if settings.openai_api_key
        else OpenAI()
    )
    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    answer = (resp.choices[0].message.content or "").strip()

    cited_labels = _parse_citation_labels(answer)
    chunk_by_label: dict[str, RetrievedChunk] = {
        f"S{i}": c for i, c in enumerate(chunks, start=1)
    }
    citations: list[Citation] = []
    raw_ids: list[str] = []
    for lab in sorted(cited_labels, key=lambda x: int(x)):
        full = f"S{lab}"
        ch = chunk_by_label.get(full)
        if ch:
            citations.append(
                Citation(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    source_path=ch.source_path,
                )
            )
            raw_ids.append(ch.chunk_id)

    return QueryResponse(
        answer=answer,
        citations=citations,
        retrieved=chunks,
        raw_citation_ids=raw_ids,
    )
