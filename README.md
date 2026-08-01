# Post-Training Quantization Trade-offs for Compact Neural Reranking in Lightweight Retrieval

> **Post-Training Quantization Trade-offs for Compact Neural Reranking in Lightweight Retrieval**  
> Yaowen Sun

## Overview

This repository contains a sanitized reproduction bundle for a compact neural
reranking measurement study. It compares BM25-only candidate ranking,
full-precision compact reranking, and dynamic int8 compact reranking on public
retrieval datasets.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   ├── public_summary.json
│   ├── formal_aggregate_metrics.csv
│   ├── formal_query_metrics.csv
│   ├── paired_int8_vs_fp32_deltas.csv
│   └── statistics_summary.json
├── figures/
│   ├── quality_cost_pareto.png
│   └── ndcg_delta_heatmap.png
├── src/
│   ├── verify_public_results.py
│   └── ranking_metrics.py
└── tests/
    └── test_public_results.py
```

## Experimental Setup

| Dimension | Values | Count |
| --- | --- | ---: |
| Datasets | `arguana`, `fiqa`, `nfcorpus`, `scifact` | 4 |
| Candidate depths | `20`, `50`, `100` | 3 |
| Methods | `bm25_no_rerank`, `fp32_reranker_cpu`, `dynamic_int8_reranker_cpu` | 3 |
| Seeds | `1`, `2`, `3`, `4`, `5` | 5 |
| Query cap | `100` per dataset-seed where qrels permit | 100 |

Row-count check:

```text
4 datasets x 3 depths x 3 methods x 5 seeds = 180 aggregate rows
4 datasets x 3 depths x 3 methods x 5 seeds x 100 queries = 18000 query rows
```

## Hardware & Environment

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

- Dynamic int8 reranking preserves nDCG@10 within the planned bounded-loss gate.
- The mean int8-vs-fp32 nDCG@10 delta is `-0.0032`.
- The mean latency ratio is `0.7774`, and the model-size ratio is `0.6450`.
- Recall@100 is unchanged because all reranking variants operate over the same first-stage candidates.

## Requirements

The verification script uses only the Python standard library. The original
matrix used the software versions listed above.

```bash
python src/verify_public_results.py
python -m unittest discover -s tests -q
```

## Citation

```bibtex
@article{sun2026quantizedrerankingtradeoffs,
  title = {Post-Training Quantization Trade-offs for Compact Neural Reranking in Lightweight Retrieval},
  author = {Sun, Yaowen},
  year = {2026}
}
```
