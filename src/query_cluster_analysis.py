#!/usr/bin/env python3
"""Reanalyze P23 quality deltas at the unique dataset-query unit."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = ("dynamic_int8_reranker_cpu", "fp32_reranker_cpu")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collapse_query_deltas(rows: list[dict[str, str]]) -> dict[str, list[dict[str, float | str]]]:
    indexed = {
        (row["dataset"], int(row["depth"]), int(row["seed"]), row["query_id"], row["method"]): row
        for row in rows
        if row["method"] in METHODS
    }
    repeated: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (dataset, depth, seed, query_id, method), int8 in indexed.items():
        if method != "dynamic_int8_reranker_cpu":
            continue
        fp32 = indexed[(dataset, depth, seed, query_id, "fp32_reranker_cpu")]
        repeated[(dataset, query_id)].append(
            (
                float(int8["ndcg@10"]) - float(fp32["ndcg@10"]),
                float(int8["mrr@10"]) - float(fp32["mrr@10"]),
            )
        )

    collapsed: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for (dataset, query_id), values in sorted(repeated.items()):
        collapsed[dataset].append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "observation_count": len(values),
                "ndcg@10_delta": statistics.mean(value[0] for value in values),
                "mrr@10_delta": statistics.mean(value[1] for value in values),
            }
        )
    return dict(collapsed)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[index]


def summarize_query_deltas(
    collapsed: dict[str, list[dict[str, float | str]]],
    bootstrap_samples: int = 10000,
    seed: int = 20260807,
) -> dict[str, dict[str, float | int | dict[str, float]]]:
    metrics = ("ndcg@10", "mrr@10")
    summary: dict[str, dict[str, float | int | dict[str, float]]] = {}
    for metric in metrics:
        key = f"{metric}_delta"
        dataset_means = {
            dataset: statistics.mean(float(row[key]) for row in rows)
            for dataset, rows in sorted(collapsed.items())
        }
        result: dict[str, float | int | dict[str, float]] = {
            "mean": statistics.mean(dataset_means.values()),
            "unique_dataset_query_units": sum(len(rows) for rows in collapsed.values()),
            "dataset_means": dataset_means,
        }
        if bootstrap_samples:
            rng = random.Random(seed)
            draws = []
            for _ in range(bootstrap_samples):
                per_dataset = []
                for dataset, rows in sorted(collapsed.items()):
                    values = [float(row[key]) for row in rows]
                    sampled = [values[rng.randrange(len(values))] for _ in values]
                    per_dataset.append(statistics.mean(sampled))
                draws.append(statistics.mean(per_dataset))
            result["bootstrap_samples"] = bootstrap_samples
            result["ci_low"] = percentile(draws, 0.025)
            result["ci_high"] = percentile(draws, 0.975)
        summary[metric] = result
    return summary


def write_outputs(collapsed, summary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat = [row for dataset in sorted(collapsed) for row in collapsed[dataset]]
    with (output_dir / "query_level_collapsed_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (output_dir / "query_cluster_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    collapsed = collapse_query_deltas(read_csv(args.query_csv))
    summary = summarize_query_deltas(collapsed, args.bootstrap_samples)
    write_outputs(collapsed, summary, args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
