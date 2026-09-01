#!/usr/bin/env python3
"""Regenerate active quantized-reranking figures from archived evidence."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FIGURE_STEMS = ("quality_cost_pareto", "latency_vs_depth", "ndcg_delta_heatmap")
METHOD_LABELS = {
    "bm25_no_rerank": "BM25",
    "fp32_reranker_cpu": "FP32",
    "dynamic_int8_reranker_cpu": "INT8",
}
DATASET_LABELS = {
    "arguana": "ArguAna",
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
}
COLORS = {"BM25": "#2f855a", "FP32": "#2563a6", "INT8": "#d97706"}
MARKERS = {"BM25": "s", "FP32": "o", "INT8": "^"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError(names)


def cutoff(row: dict[str, str]) -> int:
    return int(float(field(row, "maximum_candidate_cutoff", "depth")))


def latency(row: dict[str, str]) -> float:
    return float(field(row, "end_to_end_latency_s_per_query", "mean_latency_s_per_query", "latency_s_per_query"))


def ndcg(row: dict[str, str]) -> float:
    return float(field(row, "mean_ndcg@10", "ndcg@10"))


def latency_by_cutoff(rows: list[dict[str, str]]) -> dict[str, dict[int, float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(METHOD_LABELS[row["method"]], cutoff(row))].append(latency(row))
    values: dict[str, dict[int, float]] = defaultdict(dict)
    for (method, candidate_cutoff), samples in grouped.items():
        values[method][candidate_cutoff] = statistics.mean(samples)
    return dict(values)


def delta_grid(rows: list[dict[str, str]]) -> tuple[list[str], list[int], list[list[float]]]:
    datasets = ["ArguAna", "FiQA", "NFCorpus", "SciFact"]
    depths = [20, 50, 100]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(DATASET_LABELS[row["dataset"]], cutoff(row))].append(
            float(row["int8_minus_fp32_ndcg@10"])
        )
    values = [[statistics.mean(grouped[(dataset, depth)]) for depth in depths] for dataset in datasets]
    return datasets, depths, values


def apply_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_both(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")


def render_figures(summary_csv: Path, paired_csv: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    apply_style(plt)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(summary_csv)
    paired_rows = read_csv(paired_csv)

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for method in ("BM25", "FP32", "INT8"):
        rows = [row for row in summary_rows if METHOD_LABELS[row["method"]] == method]
        ax.scatter(
            [latency(row) for row in rows],
            [ndcg(row) for row in rows],
            s=20,
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.88,
            label=method,
        )
    ax.set_xlabel("Latency (s/query)")
    ax.set_ylabel("nDCG@10")
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.legend(frameon=True, framealpha=0.94, borderpad=0.3, handletextpad=0.35)
    fig.tight_layout()
    save_both(fig, output_dir, "quality_cost_pareto")
    plt.close(fig)

    latency_values = latency_by_cutoff(summary_rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    for method in ("BM25", "FP32", "INT8"):
        depths = sorted(latency_values[method])
        ax.plot(
            depths,
            [latency_values[method][depth] for depth in depths],
            marker=MARKERS[method],
            markersize=4.0,
            color=COLORS[method],
            label=method,
        )
    ax.set_xlabel("Maximum candidate cutoff")
    ax.set_ylabel("Latency (s/query)")
    ax.set_xticks([20, 50, 100])
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.legend(frameon=True, framealpha=0.94, borderpad=0.3, handletextpad=0.35)
    fig.tight_layout()
    save_both(fig, output_dir, "latency_vs_depth")
    plt.close(fig)

    datasets, depths, values = delta_grid(paired_rows)
    bound = max(abs(value) for row in values for value in row)
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    image = ax.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound), aspect="auto")
    ax.set_xticks(range(len(depths)), depths)
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_xlabel("Maximum candidate cutoff")
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:+.3f}", ha="center", va="center", fontsize=7, color="black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("INT8 - FP32 nDCG@10", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_both(fig, output_dir, "ndcg_delta_heatmap")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_figures(args.summary_csv, args.paired_csv, args.output_dir)
    print(f"PASS: generated {len(FIGURE_STEMS) * 2} files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
