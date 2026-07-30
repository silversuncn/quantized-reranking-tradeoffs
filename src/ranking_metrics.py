"""Small ranking metrics used for the study readiness checks."""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def validate_ranked_doc_ids(ranked_doc_ids: Iterable[str]) -> list[str]:
    doc_ids = list(ranked_doc_ids)
    if any(not doc_id for doc_id in doc_ids):
        raise ValueError("ranked_doc_ids must not contain empty document ids")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("ranked_doc_ids must be unique")
    return doc_ids


def _positive_relevances(qrels: Mapping[str, float]) -> list[float]:
    return [float(rel) for rel in qrels.values() if float(rel) > 0.0]


def _dcg(relevances: Iterable[float]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(index + 2.0) for index, rel in enumerate(relevances))


def ndcg_at_k(ranked_doc_ids: Iterable[str], qrels: Mapping[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    doc_ids = validate_ranked_doc_ids(ranked_doc_ids)[:k]
    gains = [float(qrels.get(doc_id, 0.0)) for doc_id in doc_ids]
    ideal = sorted(_positive_relevances(qrels), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def mrr_at_k(ranked_doc_ids: Iterable[str], qrels: Mapping[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    for index, doc_id in enumerate(validate_ranked_doc_ids(ranked_doc_ids)[:k], start=1):
        if float(qrels.get(doc_id, 0.0)) > 0.0:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_doc_ids: Iterable[str], qrels: Mapping[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_doc_ids = {doc_id for doc_id, rel in qrels.items() if float(rel) > 0.0}
    if not relevant_doc_ids:
        return 0.0
    retrieved = set(validate_ranked_doc_ids(ranked_doc_ids)[:k])
    return len(retrieved & relevant_doc_ids) / len(relevant_doc_ids)
