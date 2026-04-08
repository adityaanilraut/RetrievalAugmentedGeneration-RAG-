from __future__ import annotations

from openai import OpenAI

from rag_system.config import get_settings


def get_client() -> OpenAI:
    s = get_settings()
    if s.openai_api_key:
        return OpenAI(api_key=s.openai_api_key)
    return OpenAI()


def embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    m = model or s.openai_embed_model
    client = get_client()
    # Batch in chunks to respect token limits
    batch_size = 64
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=m)
        # Preserve order by index
        by_idx = {d.index: d.embedding for d in resp.data}
        for j in range(len(batch)):
            out.append(by_idx[j])
    return out


def embed_query(text: str, *, model: str | None = None) -> list[float]:
    return embed_texts([text], model=model)[0]
