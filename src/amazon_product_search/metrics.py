from typing import Dict, List, Optional, Set

import numpy as np

LABEL_TO_GAIN: Dict[str, float] = {
    "exact": 1.0,
    "substitute": 0.1,
    "complement": 0.01,
    "irrelevant": 0.0,
}


def compute_recall(retrieved_ids: List[str], relevant_ids: Set[str]) -> int:
    for retrieved_id in retrieved_ids:
        if retrieved_id in relevant_ids:
            return 1
    return 0


def compute_ap(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, retrieved_id in enumerate(retrieved_ids):
        if retrieved_id in relevant_ids:
            return 1 / (i + 1)
    return 0


def compute_dcg(gains: List[float]) -> float:
    result = 0.0
    for i, gain in enumerate(gains):
        result += gain / np.log2(i + 2)
    return result


def compute_ndcg(retrieved_ids: List[str], judgements: Dict[str, str]) -> Optional[float]:
    y_pred = [LABEL_TO_GAIN[judgements[doc_id]] if doc_id in judgements else 0 for doc_id in retrieved_ids]
    y_true = sorted(y_pred, reverse=True)
    idcg_val = compute_dcg(y_true)
    dcg_val = compute_dcg(y_pred)
    ndcg = dcg_val / idcg_val if idcg_val != 0 else None
    return ndcg
