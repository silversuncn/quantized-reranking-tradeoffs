#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AGGREGATE_V2 = "formal_aggregate_metrics_v2.csv"
REMOVED_ACTIVE_FILES = ("formal_aggregate_metrics.csv", "statistics_summary.json")
LEGACY_SUMMARY = "statistics_summary_LEGACY_DO_NOT_USE.json"
MAIN_RATIO_FIELD = "int8_end_to_end_latency_ratio_mean"
LEGACY_RATIO_FIELD = "int8_latency_ratio_mean"
EXPECTED_END_TO_END_RATIO = 0.787322079555077
EXPECTED_LEGACY_RERANKER_RATIO = 0.7773979139667812


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError(names)


def cutoff(row: dict[str, str]) -> int:
    return int(float(value(row, "maximum_candidate_cutoff", "depth")))


def as_float(row: dict[str, str], *names: str) -> float:
    return float(value(row, *names))


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{label} mismatch: {actual} != {expected}")


def assert_removed_active_files() -> None:
    stale = [name for name in REMOVED_ACTIVE_FILES if (DATA / name).exists()]
    if stale:
        raise AssertionError(f"stale active data files are present: {stale}")
    legacy_path = DATA / LEGACY_SUMMARY
    if legacy_path.exists() and "LEGACY_DO_NOT_USE" not in legacy_path.name:
        raise AssertionError("legacy statistics summary is not clearly marked")


def assert_public_summary_policy(summary: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    qs = summary["quality_cost_summary"]
    if not isinstance(qs, dict):
        raise AssertionError("quality_cost_summary is not an object")
    if qs.get("main_latency_ratio_field") != MAIN_RATIO_FIELD:
        raise AssertionError("main latency ratio field does not point to the end-to-end ratio")
    assert_close(
        float(qs[MAIN_RATIO_FIELD]),
        EXPECTED_END_TO_END_RATIO,
        "summary end-to-end ratio mean",
    )
    if LEGACY_RATIO_FIELD in qs:
        assert_close(
            float(qs[LEGACY_RATIO_FIELD]),
            EXPECTED_LEGACY_RERANKER_RATIO,
            "legacy reranker-only ratio mean",
        )
    for metric in ("ndcg@10_delta", "mrr@10_delta"):
        metric_ci = summary["bootstrap_ci"][metric]
        if not isinstance(metric_ci, dict):
            raise AssertionError(f"{metric} CI is not an object")
        if float(metric_ci["ci_high"]) < 0.0:
            raise AssertionError(f"{metric} active CI is fully negative; legacy statistics may have leaked")
    return qs, summary["bootstrap_ci"]


def build_report() -> dict[str, object]:
    assert_removed_active_files()
    summary = json.loads((DATA / "public_summary.json").read_text(encoding="utf-8"))
    qs, _ci = assert_public_summary_policy(summary)
    candidate_summary = json.loads((DATA / "candidate_latency_summary_v2.json").read_text(encoding="utf-8"))
    aggregate = rows(AGGREGATE_V2)
    query = rows("formal_query_metrics.csv")
    distribution = rows("candidate_depth_distribution_v2.csv")
    corrected = rows("corrected_throughput_v2.csv")
    latency_boundary = rows("latency_boundary_v2.csv")
    ratios = rows("end_to_end_int8_fp32_ratios_v2.csv")
    collapsed = rows("query_level_collapsed_deltas.csv")
    query_summary = json.loads((DATA / "query_cluster_summary.json").read_text(encoding="utf-8"))
    agg_keys = {(r["dataset"], cutoff(r), r["seed"], r["method"]) for r in aggregate}
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
    if sorted({cutoff(row) for row in aggregate}) != [20, 50, 100]:
        raise AssertionError("candidate cutoff grid mismatch")
    for required in (
        "maximum_candidate_cutoff",
        "actual_candidate_count",
        "corrected_throughput",
        "retrieval_latency_s_per_query",
        "end_to_end_latency_s_per_query",
        "method_latency_boundary",
    ):
        if required not in aggregate[0]:
            raise AssertionError(f"missing v2 aggregate field: {required}")
    if len(distribution) != 12:
        raise AssertionError(f"candidate distribution rows {len(distribution)} != 12")
    if len(corrected) != 180:
        raise AssertionError(f"corrected throughput rows {len(corrected)} != 180")
    if len(latency_boundary) != 180:
        raise AssertionError(f"latency boundary rows {len(latency_boundary)} != 180")
    if len(ratios) != 60:
        raise AssertionError(f"ratio rows {len(ratios)} != 60")
    nfc_depth20 = next(
        row
        for row in distribution
        if row["dataset"] == "nfcorpus" and int(row["maximum_candidate_cutoff"]) == 20
    )
    assert_close(float(nfc_depth20["mean_actual_candidate_count"]), 15.308, "nfcorpus cutoff20 mean candidates")
    if int(nfc_depth20["zero_candidate_count"]) != 39:
        raise AssertionError("nfcorpus cutoff20 zero-candidate count mismatch")
    if nfc_depth20["all_actual_equals_nominal"] != "False":
        raise AssertionError("nfcorpus cutoff20 nominal-deviation flag mismatch")
    ratio_mean = statistics.mean(float(row["int8_fp32_end_to_end_latency_ratio"]) for row in ratios)
    assert_close(ratio_mean, EXPECTED_END_TO_END_RATIO, "INT8/FP32 end-to-end ratio mean")
    assert_close(float(qs[MAIN_RATIO_FIELD]), ratio_mean, "summary end-to-end ratio mean")
    if candidate_summary["query_units"] != 6000:
        raise AssertionError("candidate summary query-unit count mismatch")
    if summary["revision_v2_20260830"]["nfcorpus_cutoff20"]["zero_candidate_count"] != 39:
        raise AssertionError("public summary nfcorpus cutoff20 mismatch")
    for row in aggregate:
        actual_candidates = as_float(row, "actual_candidate_count")
        if actual_candidates < 0 or actual_candidates > cutoff(row):
            raise AssertionError("actual candidate count outside configured cutoff")
        retrieval_latency = as_float(row, "retrieval_latency_s_per_query")
        end_to_end_latency = as_float(row, "end_to_end_latency_s_per_query")
        reranker_text = row.get("reranker_latency_s_per_query", "")
        reranker_latency = float(reranker_text) if reranker_text else 0.0
        assert_close(retrieval_latency + reranker_latency, end_to_end_latency, "latency decomposition", tol=1e-10)
        expected_boundary = "retrieval_only" if row["method"] == "bm25_no_rerank" else "retrieval_plus_reranker"
        if row["method_latency_boundary"] != expected_boundary:
            raise AssertionError(f"latency boundary mismatch for {row['method']}")
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
        "query_seed_units": candidate_summary["query_units"],
        "candidate_distribution_rows": len(distribution),
        "corrected_throughput_rows": len(corrected),
        "latency_boundary_rows": len(latency_boundary),
        "ratio_rows": len(ratios),
        "nfcorpus_cutoff20_mean_actual_candidate_count": float(nfc_depth20["mean_actual_candidate_count"]),
        "nfcorpus_cutoff20_zero_candidate_count": int(nfc_depth20["zero_candidate_count"]),
        "int8_fp32_end_to_end_ratio_mean": ratio_mean,
        "main_latency_ratio_field": qs["main_latency_ratio_field"],
        "legacy_reranker_only_ratio_mean": qs.get(LEGACY_RATIO_FIELD),
        "aggregate_file": AGGREGATE_V2,
        "legacy_statistics_file": LEGACY_SUMMARY if (DATA / LEGACY_SUMMARY).exists() else None,
        "unique_query_units": len(collapsed),
        "datasets": sorted({r["dataset"] for r in aggregate}),
        "methods": sorted({r["method"] for r in aggregate}),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
