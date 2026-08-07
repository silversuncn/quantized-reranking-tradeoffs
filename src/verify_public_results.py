#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_report() -> dict[str, object]:
    summary = json.loads((DATA / "public_summary.json").read_text(encoding="utf-8"))
    aggregate = rows("formal_aggregate_metrics.csv")
    query = rows("formal_query_metrics.csv")
    collapsed = rows("query_level_collapsed_deltas.csv")
    query_summary = json.loads((DATA / "query_cluster_summary.json").read_text(encoding="utf-8"))
    agg_keys = {(r["dataset"], r["depth"], r["seed"], r["method"]) for r in aggregate}
    query_keys = {(r["dataset"], r["depth"], r["seed"], r["method"], r["query_id"]) for r in query}
    if len(aggregate) != 180:
        raise AssertionError(f"aggregate rows {len(aggregate)} != 180")
    if len(agg_keys) != 180:
        raise AssertionError("duplicate aggregate keys")
    if len(query) != 18000:
        raise AssertionError(f"query rows {len(query)} != 18000")
    if len(query_keys) != 18000:
        raise AssertionError("duplicate query keys")
    if summary["row_counts"]["aggregate_rows"] != 180:
        raise AssertionError("summary aggregate row mismatch")
    if summary["row_counts"]["query_rows"] != 18000:
        raise AssertionError("summary query row mismatch")
    if summary.get("query_cap") != 100:
        raise AssertionError("summary query cap mismatch")
    if len(collapsed) != 1335:
        raise AssertionError(f"collapsed query units {len(collapsed)} != 1335")
    collapsed_keys = {(row["dataset"], row["query_id"]) for row in collapsed}
    if len(collapsed_keys) != 1335:
        raise AssertionError("duplicate collapsed dataset-query keys")
    expected_analysis = {
        "ndcg@10": (-0.0019086548384222395, -0.0047795560038684515, 0.00100957907113971),
        "mrr@10": (-0.001991521264799155, -0.006067751061335776, 0.0020882221551068564),
    }
    for metric, (mean, low, high) in expected_analysis.items():
        actual = query_summary[metric]
        if actual["unique_dataset_query_units"] != 1335:
            raise AssertionError(f"{metric} unique-unit count mismatch")
        for key, expected in (("mean", mean), ("ci_low", low), ("ci_high", high)):
            if abs(float(actual[key]) - expected) > 1e-12:
                raise AssertionError(f"{metric} {key} mismatch")
    return {
        "status": "PASS",
        "aggregate_rows": len(aggregate),
        "query_rows": len(query),
        "unique_query_units": len(collapsed),
        "datasets": sorted({r["dataset"] for r in aggregate}),
        "methods": sorted({r["method"] for r in aggregate}),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
