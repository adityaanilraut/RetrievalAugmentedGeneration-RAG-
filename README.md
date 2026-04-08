# RAG System (Python + OpenAI + PostgreSQL pgvector)

Backend-only RAG: ingest PDF/TXT/Markdown, chunk, embed with OpenAI, store vectors in **PostgreSQL + pgvector**, retrieve top‑k chunks, generate **grounded answers with citations**. Includes **FastAPI**, **CLI**, and **BEIR-style dense retrieval benchmarks** (SciFact, FiQA) with **Recall@k**, **NDCG@k**, **MAP@k**, **P@k**, and **MRR** (evaluation does not use the database).

## Architecture

```mermaid
flowchart LR
  Docs[Local documents] --> Ingest[Extract and chunk]
  Ingest --> Embed[OpenAI embeddings]
  Embed --> PG[(PostgreSQL pgvector)]
  Q[Question] --> QEmb[Query embedding]
  QEmb --> Search[Top-k similarity]
  PG --> Search
  Search --> Gen[OpenAI chat with sources]
  Gen --> Out[Answer plus citations]
```

## Prerequisites

- Python 3.11+
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension enabled (local install, managed service, or any host you can reach)
- `OPENAI_API_KEY` for embeddings and chat

## Quick start

### 1. Database

Create a database and enable the extension (as a superuser or where allowed):

```sql
CREATE DATABASE rag;
\c rag
CREATE EXTENSION IF NOT EXISTS vector;
```

Point `DATABASE_URL` at that database (see [Configuration](#configuration)).

### 2. Environment

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and DATABASE_URL if needed
```

### 3. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
# Optional: charts from saved JSON (no API calls when plotting only)
pip install -e ".[benchmark-plots]"
```

### 4. Initialize schema

Schema is applied automatically on API startup or when you run CLI commands that touch the DB. You can also run:

```bash
python -c "from rag_system.db.pg import init_db; init_db()"
```

### 5. Ingest local documents

```bash
rag ingest data/docs
```

### 6. Query (CLI)

```bash
rag query "What endpoints does the API expose?" --top-k 5
rag query "What is RAG?" --json
```

### 7. Run the API

```bash
rag serve --host 127.0.0.1 --port 8000
```

- `GET /health` — DB connectivity
- `POST /ingest` — JSON `{"path": "data/docs"}`
- `POST /query` — JSON `{"question": "...", "top_k": 5}`

API base URL: `http://127.0.0.1:8000` (adjust host/port if you change `rag serve` options).

## Benchmarks and results

Dense retrieval benchmarks download a [BEIR](https://github.com/beir-cellar/beir) dataset, embed corpus and test queries with the configured OpenAI embedding model, rank documents by **cosine similarity** in memory, and score against official qrels using **pytrec_eval** (via BEIR’s `EvaluateRetrieval`). **Scores depend on the embedding model, caps (`--max-corpus` / `--max-queries`), and API randomness.** The JSON under `docs/benchmarks/sample_*.json` is an illustrative snapshot for README figures; replace it with your own `--json-out` to report real runs.

### Benchmark pipeline

```mermaid
flowchart LR
  subgraph beirEval [BEIR dense eval]
    DL[Download dataset]
    EmbC[Embed corpus]
    EmbQ[Embed queries]
    Rank[Cosine similarity rank]
    Met[NDCG MAP Recall P MRR]
    DL --> EmbC
    EmbQ --> Rank
    EmbC --> Rank
    Rank --> Met
  end
  subgraph plots [Plots]
    JSON[Saved JSON]
    PNG[PNG charts]
    JSON --> PNG
  end
  Met --> JSON
```

### Metrics (at each k)

| Metric | Role |
|--------|------|
| Recall@k | Any relevant doc in the top‑k ranked list |
| NDCG@k | Graded usefulness of the ranking at k |
| MAP@k | Mean average precision at k |
| P@k | Precision at k |
| MRR | Reciprocal rank of the first relevant doc |

### Example table (illustrative sample JSON)

SciFact (`docs/benchmarks/sample_scifact.json`):

| k | Recall@k | NDCG@k | MAP@k | P@k |
|---|----------|--------|-------|-----|
| 1 | 0.62 | 0.58 | 0.58 | 0.62 |
| 3 | 0.74 | 0.68 | 0.63 | 0.35 |
| 5 | 0.79 | 0.72 | 0.65 | 0.24 |
| 10 | 0.85 | 0.76 | 0.66 | 0.14 |

FiQA (`docs/benchmarks/sample_fiqa.json`): MRR 0.36; see JSON for full k‑wise metrics.

### Charts (regenerated from JSON)

Regenerate PNGs anytime (no API calls):

```bash
pip install -e ".[benchmark-plots]"
rag plot-benchmarks docs/benchmarks/sample_scifact.json --out-dir docs/benchmarks --stem sample_scifact
rag plot-benchmarks docs/benchmarks/sample_scifact.json --compare docs/benchmarks/sample_fiqa.json --labels scifact fiqa --out-dir docs/benchmarks --stem scifact_vs_fiqa
```

Single-run SciFact (Recall@k vs NDCG@k, and MAP@k vs P@k):

![SciFact Recall and NDCG](docs/benchmarks/sample_scifact_recall_ndcg.png)

![SciFact MAP and precision](docs/benchmarks/sample_scifact_map_precision.png)

SciFact vs FiQA comparison (illustrative samples):

![Recall by dataset](docs/benchmarks/scifact_vs_fiqa_recall_by_dataset.png)

![NDCG by dataset](docs/benchmarks/scifact_vs_fiqa_ndcg_by_dataset.png)

### Run benchmarks and plots

```bash
# SciFact — smaller run (fewer API calls)
rag eval-scifact --max-corpus 500 --max-queries 50 --json-out docs/benchmarks/my_scifact.json

# Any supported BEIR dataset (scifact, fiqa)
rag eval-beir --dataset fiqa --max-corpus 2000 --max-queries 100 --json-out docs/benchmarks/my_fiqa.json

# Full test split (many embedding calls; costs apply)
rag eval-beir --dataset scifact --max-corpus 0 --max-queries 0

# Plots from saved JSON only (no API calls)
rag plot-benchmarks docs/benchmarks/my_scifact.json --out-dir docs/benchmarks
rag plot-benchmarks docs/benchmarks/my_scifact.json --compare docs/benchmarks/my_fiqa.json --labels scifact fiqa --stem my_compare --out-dir docs/benchmarks
```

Options:

- `--max-corpus 0` — use all documents in the split (after download)
- `--max-queries 0` — use all queries that have ≥1 relevant document in the embedded corpus
- `--k 1 3 5 10` — cutoffs for NDCG, MAP, Recall, P, etc.

Datasets are cached under `./datasets/` on first run.

## Configuration

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for embeddings and generation |
| `OPENAI_CHAT_MODEL` | Default: `gpt-5-mini` |
| `OPENAI_EMBED_MODEL` | Default: `text-embedding-3-small` (1536-dim) |
| `DATABASE_URL` | psycopg3 URL, e.g. `postgresql://user:pass@localhost:5432/rag` |

If you change embedding model dimensions, update `rag_system/db/schema.sql` (`vector(...)`) to match.

## Project layout

- `rag_system/core/` — ingest, chunking, embeddings, retrieval, generation
- `rag_system/db/` — PostgreSQL + pgvector schema and connection
- `rag_system/api/` — FastAPI app
- `rag_system/evaluation/` — BEIR dense benchmarks and plotting helpers
- `docs/benchmarks/` — sample eval JSON and chart images
- `data/docs/` — sample files for local testing
- `old_code/` — legacy LangChain / notebook experiment (`rag.ipynb`); not used by the current package

## License

MIT — see [`LICENSE`](LICENSE).
