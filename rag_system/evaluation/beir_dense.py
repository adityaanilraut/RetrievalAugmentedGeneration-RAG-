from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

from rag_system.core.embeddings import embed_texts
from rag_system.models import EvalBeirResult

ALLOWED_DATASETS: frozenset[str] = frozenset({"scifact", "fiqa"})


def _doc_text(entry: object) -> str:
    if isinstance(entry, dict):
        title = (entry.get("title") or "").strip()
        text = (entry.get("text") or "").strip()
    else:
        title = (getattr(entry, "title", None) or "").strip()
        text = (getattr(entry, "text", None) or "").strip()
    return f"{title}\n{text}".strip()


def _relevant_docs(qrel: dict[str, int]) -> set[str]:
    return {d for d, score in qrel.items() if float(score) > 0}


def _mean_reciprocal_rank(
    qrels: dict[str, dict[str, int]],
    ranked_by_query: dict[str, list[str]],
) -> float:
    s = 0.0
    den = 0
    for qid, rel in qrels.items():
        rel_docs = _relevant_docs(rel)
        if not rel_docs:
            continue
        den += 1
        ranked = ranked_by_query.get(qid, [])
        rr = 0.0
        for i, did in enumerate(ranked, start=1):
            if did in rel_docs:
                rr = 1.0 / i
                break
        s += rr
    return s / den if den else 0.0


def run_beir_dense_benchmark(
    *,
    dataset: str,
    max_corpus: int = 0,
    max_queries: int = 0,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    data_dir: str = "datasets",
) -> EvalBeirResult:
    """
    Dense retrieval on a BEIR dataset: embed corpus + queries with OpenAI, rank by cosine similarity.
    Does not use PostgreSQL. Metrics include NDCG@k, MAP@k, P@k, Recall@k (BEIR/pytrec_eval), and MRR.
    """
    ds = dataset.lower().strip()
    if ds not in ALLOWED_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(ALLOWED_DATASETS)}, got {dataset!r}")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds}.zip"
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = beir_util.download_and_unzip(url, str(root))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    doc_ids = sorted(corpus.keys())
    if max_corpus and max_corpus < len(doc_ids):
        doc_ids = doc_ids[:max_corpus]
    doc_set = set(doc_ids)
    sub_corpus = {k: corpus[k] for k in doc_ids}

    qrels_f: dict[str, dict[str, int]] = {}
    for qid, rel in qrels.items():
        rel_in = {d: s for d, s in rel.items() if d in doc_set and s > 0}
        if rel_in:
            qrels_f[qid] = rel_in

    qids = sorted(qrels_f.keys())
    if max_queries and max_queries < len(qids):
        qids = qids[:max_queries]
        qrels_f = {q: qrels_f[q] for q in qids}

    if not qids:
        return EvalBeirResult(
            dataset=ds,
            recall_at_k={f"Recall@{k}": 0.0 for k in k_values},
            ndcg_at_k={f"NDCG@{k}": 0.0 for k in k_values},
            map_at_k={f"MAP@{k}": 0.0 for k in k_values},
            precision_at_k={f"P@{k}": 0.0 for k in k_values},
            mrr=0.0,
            num_queries=0,
            notes="No queries left after filtering qrels to embedded corpus; increase --max-corpus.",
            extra={"corpus_size": len(doc_ids), "data_path": os.path.abspath(data_path)},
        )

    texts = [_doc_text(sub_corpus[d]) for d in doc_ids]
    doc_embs = embed_texts(texts)
    E = np.asarray(doc_embs, dtype=np.float64)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    results_scores: dict[str, dict[str, float]] = {}
    ranked_by_query: dict[str, list[str]] = {}
    q_texts = [queries[qid] for qid in qids]
    q_embs = embed_texts(q_texts)

    for qid, qv in tqdm(
        zip(qids, q_embs, strict=True),
        total=len(qids),
        desc=f"BEIR {ds} queries",
    ):
        q = np.asarray(qv, dtype=np.float64)
        qn = q / (np.linalg.norm(q) + 1e-12)
        scores = E_norm @ qn
        results_scores[qid] = {doc_ids[i]: float(scores[i]) for i in range(len(doc_ids))}
        order = np.argsort(-scores)
        ranked_by_query[qid] = [doc_ids[i] for i in order.tolist()]

    # BEIR evaluate may mutate results; pass deep-ish copy of inner score dicts
    to_eval = {qid: dict(scores) for qid, scores in results_scores.items()}
    log = logging.getLogger("beir.retrieval.evaluation")
    prev = log.level
    log.setLevel(logging.WARNING)
    try:
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels_f,
            to_eval,
            list(k_values),
            ignore_identical_ids=True,
        )
    finally:
        log.setLevel(prev)

    mrr = _mean_reciprocal_rank(qrels_f, ranked_by_query)

    return EvalBeirResult(
        dataset=ds,
        recall_at_k=dict(recall),
        ndcg_at_k=dict(ndcg),
        map_at_k=dict(_map),
        precision_at_k=dict(precision),
        mrr=mrr,
        num_queries=len(qids),
        notes=(
            f"Dense OpenAI embedding retrieval vs BEIR {ds} test qrels; "
            "corpus may be truncated with --max-corpus."
        ),
        extra={
            "corpus_size": len(doc_ids),
            "data_path": os.path.abspath(data_path),
        },
    )
