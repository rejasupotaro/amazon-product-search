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
    return round(len([x for x in xs if x == 0]) / len(xs), 4)


def compute_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> Optional[float]:
    if not relevant_ids:
        return None
    return round(len(set(retrieved_ids) & relevant_ids) / len(relevant_ids), 4)


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
    return round(gain / num_relevant_docs, 4)


def compute_dcg(gains: list[float]) -> float:
    result = 0.0
    for i, gain in enumerate(gains):
        result += gain / np.log2(i + 2)
    return result


def compute_ndcg(retrieved_ids: list[str], judgements: dict[str, str], prime: bool = False) -> Optional[float]:
    """Compute Normalized Discounted Cumulative Gain (NDCG) based on the given relevance judgements.

    NDCG' is a variant of NDCG that calculates the score using only annotated documents.

    Args:
        retrieved_ids (list[str]): Document IDs retrieved from a search engine.
        judgements (dict[str, str]): An annotation set for the query.
        prime (bool, optional): True to skip unseen documents. Defaults to False.

    Returns:
        Optional[float]: The computed score or None.
    """
    if prime:
        y_pred = [LABEL_TO_GAIN[judgements[doc_id]] for doc_id in retrieved_ids if doc_id in judgements]
    else:
        y_pred = [LABEL_TO_GAIN[judgements[doc_id]] if doc_id in judgements else 0 for doc_id in retrieved_ids]
    y_true = sorted([LABEL_TO_GAIN[label] for label in judgements.values()], reverse=True)
    idcg_val = compute_dcg(y_true)
    dcg_val = compute_dcg(y_pred)
    ndcg = round(dcg_val / idcg_val, 4) if idcg_val != 0 else None
    return ndcg


def compute_cosine_similarity(query_vector: np.ndarray, product_vectors: np.ndarray) -> np.ndarray:
    numerator = np.dot(query_vector, product_vectors.T)
    denominator = (np.linalg.norm(query_vector) * np.linalg.norm(product_vectors, axis=1))
    return numerator / denominator
