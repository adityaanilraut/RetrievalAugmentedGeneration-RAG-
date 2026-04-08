from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from rag_system.config import get_settings
from rag_system.core.generation import generate_answer
from rag_system.core.ingest import ingest_path
from rag_system.core.retrieval import retrieve_chunks
from rag_system.db.pg import init_db
from rag_system.evaluation.beir_dense import ALLOWED_DATASETS, run_beir_dense_benchmark


def _print_eval_result(result: object, json_out: str | None) -> None:
    text = json.dumps(result.model_dump(), indent=2)
    print(text)
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            f.write(text)


def cmd_ingest(args: argparse.Namespace) -> int:
    init_db()
    files, chunks = ingest_path(args.path)
    print(f"OK: {files} file(s), {chunks} chunk(s) written.")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    init_db()
    chunks = retrieve_chunks(args.question, top_k=args.top_k)
    out = generate_answer(args.question, chunks)
    if args.json:
        print(json.dumps(out.model_dump(), indent=2))
    else:
        print(out.answer)
        if out.citations:
            print("\nCitations:")
            for c in out.citations:
                print(f"  - {c.chunk_id} ({c.source_path})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ.setdefault("UVICORN_HOST", args.host)
    os.environ.setdefault("UVICORN_PORT", str(args.port))
    uvicorn.run(
        "rag_system.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_eval_beir(args: argparse.Namespace) -> int:
    result = run_beir_dense_benchmark(
        dataset=args.dataset,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        k_values=tuple(args.k),
        data_dir=args.data_dir,
    )
    _print_eval_result(result, args.json_out)
    return 0


def cmd_eval_scifact(args: argparse.Namespace) -> int:
    result = run_beir_dense_benchmark(
        dataset="scifact",
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        k_values=tuple(args.k),
        data_dir=args.data_dir,
    )
    _print_eval_result(result, args.json_out)
    return 0


def cmd_plot_benchmarks(args: argparse.Namespace) -> int:
    from rag_system.evaluation.plot_results import main as plot_main

    pieces: list[str] = [str(args.input_path), "--out-dir", str(args.out_dir)]
    if args.compare:
        pieces.extend(["--compare", str(args.compare)])
    if args.labels:
        pieces.extend(["--labels", *args.labels])
    if args.stem:
        pieces.extend(["--stem", args.stem])
    return plot_main(pieces)


def _add_beir_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-corpus", type=int, default=0, help="0 = all")
    p.add_argument("--max-queries", type=int, default=0, help="0 = all")
    p.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="k cutoffs for NDCG, MAP, Recall, P, etc.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Directory to download/cache BEIR datasets",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Also write results JSON to this file (for plotting)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag", description="RAG CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDF/TXT/MD from path")
    p_ingest.add_argument("path", help="File or directory")
    p_ingest.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="Run retrieval + grounded answer")
    p_q.add_argument("question", help="Question text")
    p_q.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_q.add_argument("--json", action="store_true", help="JSON output")
    p_q.set_defaults(func=cmd_query)

    p_s = sub.add_parser("serve", help="Start FastAPI server")
    p_s.add_argument("--host", default="127.0.0.1")
    p_s.add_argument("--port", type=int, default=8000)
    p_s.add_argument("--reload", action="store_true")
    p_s.set_defaults(func=cmd_serve)

    p_beir = sub.add_parser(
        "eval-beir",
        help="BEIR dense retrieval benchmark (SciFact, FiQA, ...)",
    )
    p_beir.add_argument(
        "--dataset",
        choices=sorted(ALLOWED_DATASETS),
        required=True,
        help="BEIR dataset name",
    )
    _add_beir_eval_args(p_beir)
    p_beir.set_defaults(func=cmd_eval_beir)

    p_e = sub.add_parser(
        "eval-scifact",
        help="Same as: eval-beir --dataset scifact",
    )
    _add_beir_eval_args(p_e)
    p_e.set_defaults(func=cmd_eval_scifact)

    p_plot = sub.add_parser(
        "plot-benchmarks",
        help="Plot metrics from eval JSON (needs: pip install -e '.[benchmark-plots]')",
    )
    p_plot.add_argument(
        "input_path",
        type=str,
        help="JSON from rag eval-beir --json-out",
    )
    p_plot.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Second JSON for dataset comparison charts",
    )
    p_plot.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels (1 or 2 runs)",
    )
    p_plot.add_argument(
        "--out-dir",
        type=str,
        default="docs/benchmarks",
        help="Output directory for PNG files",
    )
    p_plot.add_argument(
        "--stem",
        type=str,
        default="",
        help="Filename stem for outputs",
    )
    p_plot.set_defaults(func=cmd_plot_benchmarks)

    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    _ = get_settings()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
