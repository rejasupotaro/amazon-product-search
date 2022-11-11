from typing import Optional

import numpy as np

LABEL_TO_GAIN: dict[str, float] = {
    "E": 1.0,
    "S": 0.1,
    "C": 0.01,
    "I": 0.0,
}


def compute_zero_hit_rate(xs: list[int]) -> Optional[float]:
    if len(xs) == 0:
        return None
    return len([x for x in xs if x == 0]) / len(xs)


def compute_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> Optional[float]:
    if not relevant_ids:
        return None
    return len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)


def compute_ap(retrieved_ids: list[str], relevant_ids: set[str]) -> Optional[float]:
    if not retrieved_ids or not relevant_ids:
        return None

    gain = 0.0
    num_relevant_docs = 0
    for i, retrieved_id in enumerate(retrieved_ids):
        if retrieved_id in relevant_ids:
            num_relevant_docs += 1
            gain += num_relevant_docs / (i + 1)
    if num_relevant_docs == 0:
        return None
    return gain / num_relevant_docs


def compute_dcg(gains: list[float]) -> float:
    result = 0.0
    for i, gain in enumerate(gains):
        result += gain / np.log2(i + 2)
    return result


def compute_ndcg(retrieved_ids: list[str], judgements: dict[str, str]) -> Optional[float]:
    y_pred = [LABEL_TO_GAIN[judgements[doc_id]] if doc_id in judgements else 0 for doc_id in retrieved_ids]
    y_true = sorted(y_pred, reverse=True)
    idcg_val = compute_dcg(y_true)
    dcg_val = compute_dcg(y_pred)
    ndcg = dcg_val / idcg_val if idcg_val != 0 else None
    return ndcg
