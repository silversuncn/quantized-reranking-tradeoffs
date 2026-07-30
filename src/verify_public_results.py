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
    agg_keys = {(r["dataset"], r["depth"], r["seed"], r["method"]) for r in aggregate}
    query_keys = {(r["dataset"], r["depth"], r["seed"], r["method"], r["query_id"]) for r in query}
    if len(aggregate) != 180:
        raise AssertionError(f"aggregate rows {len(aggregate)} != 180")
    if len(agg_keys) != 180:
        raise AssertionError("duplicate aggregate keys")
    if len(query) != 4320:
        raise AssertionError(f"query rows {len(query)} != 4320")
    if len(query_keys) != 4320:
        raise AssertionError("duplicate query keys")
    if summary["row_counts"]["aggregate_rows"] != 180:
        raise AssertionError("summary aggregate row mismatch")
    gate = summary["quality_cost_summary"]["gate"]
    if gate["status"] != "PASS":
        raise AssertionError("quality-cost gate did not pass")
    return {
        "status": "PASS",
        "aggregate_rows": len(aggregate),
        "query_rows": len(query),
        "datasets": sorted({r["dataset"] for r in aggregate}),
        "methods": sorted({r["method"] for r in aggregate}),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
