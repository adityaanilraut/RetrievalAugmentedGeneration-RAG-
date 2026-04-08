from __future__ import annotations

from rag_system.evaluation.beir_dense import run_beir_dense_benchmark
from rag_system.models import EvalSciFactResult


def run_scifact_benchmark(
    *,
    max_corpus: int = 0,
    max_queries: int = 0,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    data_dir: str = "datasets",
) -> EvalSciFactResult:
    """
    Dense retrieval on BEIR SciFact: embed corpus + queries with OpenAI, rank by cosine similarity.
    Does not use PostgreSQL (pure embedding retrieval for benchmarking).
    """
    return run_beir_dense_benchmark(
        dataset="scifact",
        max_corpus=max_corpus,
        max_queries=max_queries,
        k_values=k_values,
        data_dir=data_dir,
    )


__all__ = ["run_scifact_benchmark"]
