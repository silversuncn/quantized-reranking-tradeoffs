# Post-Training Quantization Trade-offs for Compact Neural Reranking in Lightweight Retrieval

> Yaowen Sun

## Overview

This repository is a derived-result verification bundle for a compact neural reranking measurement study. It contains sanitized aggregate/query-level result files, verification scripts, and figure-generation scripts for checking the reported values. It does not contain the complete retrieval and model-inference pipeline.

The study compares BM25-only candidate ranking, full-precision compact reranking, and dynamic int8 compact reranking on public BEIR-style retrieval datasets.

## Repository Structure

```text
.
|-- README.md
|-- CITATION.cff
|-- LICENSE
|-- requirements.txt
|-- checksums_sha256.txt
|-- data/
|   |-- public_summary.json
|   |-- formal_aggregate_metrics_v2.csv
|   |-- formal_query_metrics.csv
|   |-- candidate_depth_distribution_v2.csv
|   |-- corrected_throughput_v2.csv
|   |-- latency_boundary_v2.csv
|   |-- end_to_end_int8_fp32_ratios_v2.csv
|   |-- candidate_latency_summary_v2.json
|   |-- paired_int8_vs_fp32_deltas.csv
|   |-- query_level_collapsed_deltas.csv
|   |-- query_cluster_summary.json
|   `-- statistics_summary_LEGACY_DO_NOT_USE.json
|-- figures/
|   |-- quality_cost_pareto.png
|   |-- quality_cost_pareto.pdf
|   |-- latency_vs_depth.png
|   |-- latency_vs_depth.pdf
|   |-- ndcg_delta_heatmap.png
|   `-- ndcg_delta_heatmap.pdf
|-- src/
|   |-- verify_public_results.py
|   |-- query_cluster_analysis.py
|   |-- plot_figures.py
|   `-- ranking_metrics.py
`-- tests/
    `-- test_public_results.py
```

`formal_aggregate_metrics_v2.csv` is the only active aggregate CSV. The unversioned `formal_aggregate_metrics.csv` and `statistics_summary.json` files are intentionally absent. The legacy statistics file is retained only because its filename explicitly marks it as non-active historical context.

## Experimental Setup

| Dimension | Values | Count |
| --- | --- | ---: |
| Datasets | `arguana`, `fiqa`, `nfcorpus`, `scifact` | 4 |
| Maximum candidate cutoffs | `20`, `50`, `100` | 3 |
| Methods | `bm25_no_rerank`, `fp32_reranker_cpu`, `dynamic_int8_reranker_cpu` | 3 |
| Seeds | `1`, `2`, `3`, `4`, `5` | 5 |
| Query cap | `100` per dataset-seed where qrels permit | 100 |

Row-count check:

```text
4 datasets x 3 maximum candidate cutoffs x 3 methods x 5 seeds = 180 aggregate rows
4 datasets x 3 maximum candidate cutoffs x 3 methods x 5 seeds x 100 queries = 18,000 query rows
```

## Hardware and Environment

| Component | Value |
| --- | --- |
| Runtime target | CPU reranking |
| Quantization | PyTorch dynamic int8 |
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Python | 3.11.15 |
| Torch | 2.11.0+cu128 |
| Transformers | 5.4.0 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |

## Key Results

- The 60 matched dataset-cutoff-seed cells give descriptive mean deltas of `-0.0032` nDCG@10 and `-0.0046` MRR@10.
- Because queries recur across depths and may overlap across seeds, inference collapses repeated records into `1,335` unique dataset-query units.
- Dataset-balanced unique-query means are `-0.0019` nDCG@10 with 95% CI `[-0.0048, 0.0010]` and `-0.0020` MRR@10 with 95% CI `[-0.0061, 0.0021]`.
- Maximum candidate cutoffs are configured caps; actual first-stage candidate counts may be smaller for sparse queries.
- At cutoff 20, NFCorpus has mean actual candidate count `15.308` and `39/500` query-seed units with zero candidates.
- At NFCorpus cutoff 20, corrected throughput means are `25135.814346079871` candidates/s for BM25, `75.291756622626` for INT8, and `59.367108608899` for FP32.
- The main latency metric is end-to-end latency: BM25 retrieval plus cross-encoder reranker inference for reranking methods.
- The main INT8/FP32 end-to-end latency ratio is `0.787322079555077`.
- The older reranker-only ratio `0.7773979139667812` is retained only as legacy component-latency context, not as the main latency result.
- The model-size ratio is `0.6450451520275926`.
- Recall@100 is unchanged because all reranking variants operate over the same first-stage candidates.

## Verify Bundled Results

The verifier checks row counts, required v2 columns, the absence of stale active files, the end-to-end latency boundary, the public summary ratio policy, unique-query confidence intervals, and selected candidate-count diagnostics.

```bash
python src/verify_public_results.py
python -m unittest discover -s tests -q
shasum -a 256 -c checksums_sha256.txt
```

## Regenerate Derived Figures

The figure script regenerates figures from the bundled derived CSVs:

```bash
python src/plot_figures.py \
  --summary-csv data/formal_aggregate_metrics_v2.csv \
  --paired-csv data/paired_int8_vs_fp32_deltas.csv \
  --output-dir figures
```

## Citation

```bibtex
@article{sun2026quantizedrerankingtradeoffs,
  title = {Post-Training Quantization Trade-offs for Compact Neural Reranking in Lightweight Retrieval},
  author = {Sun, Yaowen},
  year = {2026}
}
```

## License

This public verification bundle is released under the license included in `LICENSE`.
