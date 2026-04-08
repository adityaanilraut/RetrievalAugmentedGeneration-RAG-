from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_at_k(metrics: dict[str, float], prefix: str) -> tuple[list[int], list[float]]:
    """e.g. prefix 'Recall@' -> keys Recall@1, Recall@3 -> ks [1,3], values aligned."""
    pairs: list[tuple[int, float]] = []
    plen = len(prefix)
    for key, val in metrics.items():
        if not key.startswith(prefix):
            continue
        rest = key[plen:]
        try:
            k = int(rest)
        except ValueError:
            continue
        pairs.append((k, float(val)))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return [], []
    ks, vs = zip(*pairs, strict=True)
    return list(ks), list(vs)


def _load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_single_run(data: dict[str, Any], out_dir: Path, basename: str) -> list[Path]:
    """Bar charts: Recall@k + NDCG@k on one figure; MAP@k + P@k on another."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    recall_k, recall_v = _parse_at_k(data.get("recall_at_k") or {}, "Recall@")
    ndcg_k, ndcg_v = _parse_at_k(data.get("ndcg_at_k") or {}, "NDCG@")
    if recall_k and ndcg_k and recall_k == ndcg_k:
        x = range(len(recall_k))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - w / 2 for i in x], recall_v, width=w, label="Recall@k")
        ax.bar([i + w / 2 for i in x], ndcg_v, width=w, label="NDCG@k")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(k) for k in recall_k])
        ax.set_xlabel("k")
        ax.set_ylabel("Score")
        ds = data.get("dataset", "run")
        ax.set_title(f"Retrieval metrics ({ds})")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        p = out_dir / f"{basename}_recall_ndcg.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    map_k, map_v = _parse_at_k(data.get("map_at_k") or {}, "MAP@")
    p_k, p_v = _parse_at_k(data.get("precision_at_k") or {}, "P@")
    if map_k and p_k and map_k == p_k:
        x = range(len(map_k))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - w / 2 for i in x], map_v, width=w, label="MAP@k")
        ax.bar([i + w / 2 for i in x], p_v, width=w, label="P@k")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(k) for k in map_k])
        ax.set_xlabel("k")
        ax.set_ylabel("Score")
        ds = data.get("dataset", "run")
        ax.set_title(f"MAP & precision ({ds})")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        p = out_dir / f"{basename}_map_precision.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    return written


def plot_compare(
    runs: list[tuple[str, dict[str, Any]]],
    out_dir: Path,
    stem: str = "compare",
) -> list[Path]:
    """Grouped bars for Recall@k and NDCG@k across named runs (same k axis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if len(runs) < 2:
        return written

    # Use k from first run's recall
    k0, _ = _parse_at_k(runs[0][1].get("recall_at_k") or {}, "Recall@")
    if not k0:
        return written

    labels = [name for name, _ in runs]
    n = len(runs)
    x = range(len(k0))
    w = 0.8 / n

    for metric_key, prefix, fname in (
        ("recall_at_k", "Recall@", "recall"),
        ("ndcg_at_k", "NDCG@", "ndcg"),
    ):
        fig, ax = plt.subplots(figsize=(max(8, len(k0) * 1.2), 4))
        for i, (_, data) in enumerate(runs):
            ks, vs = _parse_at_k(data.get(metric_key) or {}, prefix)
            if ks != k0:
                continue
            offset = (i - (n - 1) / 2) * w
            ax.bar([j + offset for j in x], vs, width=w, label=labels[i])
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(k) for k in k0])
        ax.set_xlabel("k")
        ax.set_ylabel("Score")
        ax.set_title(f"{prefix.rstrip('@')} comparison")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        path = out_dir / f"{stem}_{fname}_by_dataset.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot BEIR eval JSON (matplotlib)")
    p.add_argument("input", type=Path, help="eval JSON from rag eval-beir --json-out")
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Second JSON for grouped bar comparison",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Labels for runs (1 or 2); default: dataset field or file stem",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Directory for PNG output",
    )
    p.add_argument(
        "--stem",
        type=str,
        default="",
        help="Output filename stem for single-run plots (default: input stem)",
    )
    args = p.parse_args(argv)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "matplotlib is required. Install with: pip install -e '.[benchmark-plots]'",
            file=sys.stderr,
        )
        return 1

    data_a = _load_eval(args.input)
    stem = args.stem or args.input.stem

    out: list[Path] = []
    if args.compare:
        data_b = _load_eval(args.compare)
        la = (
            args.labels[0]
            if args.labels and len(args.labels) > 0
            else data_a.get("dataset") or args.input.stem
        )
        lb = (
            args.labels[1]
            if args.labels and len(args.labels) > 1
            else data_b.get("dataset") or args.compare.stem
        )
        out.extend(
            plot_compare(
                [(la, data_a), (lb, data_b)],
                args.out_dir,
                stem=args.stem or "compare",
            )
        )
    else:
        out.extend(plot_single_run(data_a, args.out_dir, stem))

    for path in out:
        print(path)
    if not out:
        print("No figures written (missing metrics?)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
