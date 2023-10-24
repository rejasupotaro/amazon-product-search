import numpy as np
import pytest

from amazon_product_search.core.metrics import (
    compute_ap,
    compute_cosine_similarity,
    compute_iou,
    compute_ndcg,
    compute_precision,
    compute_recall,
    compute_zero_hit_rate,
)


@pytest.mark.parametrize(
    ("xs", "expected"),
    [
        ([], None),
        ([1], 0),
        ([0], 1),
        ([1, 0], 0.5),
        ([0, 1], 0.5),
    ],
)
def test_compute_zero_hit_rate(xs, expected):
    actual = compute_zero_hit_rate(xs)
    assert actual == expected


@pytest.mark.parametrize(
    ("retrieved_ids", "relevant_ids", "expected"),
    [
        ([], {}, None),
        ([1, 2], {}, None),
        ([], {1, 2}, 0),
        ([1, 2], {1, 2}, 1),
        ([1, 2], {1}, 1),
        ([1, 2], {2}, 1),
        ([1, 2], {1, 3}, 0.5),
        ([1, 2], {2, 3}, 0.5),
    ],
)
def test_compute_recall(retrieved_ids, relevant_ids, expected):
    actual = compute_recall(retrieved_ids, relevant_ids)
    assert actual == expected


@pytest.mark.parametrize(
    ("retrieved_ids", "relevant_ids", "expected"),
    [
        ([], set(), None),
        (["1"], set(), 0),
        ([], {"1"}, None),
        (["1"], {"1"}, 1),
        (["1", "2"], {"1"}, 0.5),
        (["1"], {"1", "2"}, 1),
    ],
)
def test_compute_precision(retrieved_ids, relevant_ids, expected):
    actual = compute_precision(retrieved_ids, relevant_ids)
    assert actual == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (set(), set(), (None, set(), set())),
        ({"1"}, set(), (0, set(), {"1"})),
        (set(), {"1"}, (0, set(), {"1"})),
        ({"1"}, {"1"}, (1, {"1"}, {"1"})),
        ({"1", "2"}, {"1"}, (0.5, {"1"}, {"1", "2"})),
        ({"1", "2"}, {"3"}, (0, set(), {"1", "2", "3"})),
    ],
)
def test_compute_iou(a, b, expected):
    actual = compute_iou(a, b)
    assert actual == expected


@pytest.mark.parametrize(
    ("retrieved_ids", "relevant_ids", "expected"),
    [
        ([], {}, None),
        ([1, 2, 3, 4], {}, None),
        ([], {1, 2, 3, 4}, None),
        ([1, 2, 3, 4], {1, 2, 3, 4}, 1),
        ([1, 2, 3, 4], {1, 2}, 1),
        ([1, 2, 3, 4], {2}, 0.5),
        ([1, 2, 3, 4], {2, 4}, 0.5),
        ([1, 2, 3, 4], {4}, 0.25),
    ],
)
def test_compute_ap(retrieved_ids, relevant_ids, expected):
    actual = compute_ap(retrieved_ids, relevant_ids)
    assert actual == expected


@pytest.mark.parametrize(
    ("retrieved_ids", "id_to_label", "expected"),
    [
        ([], {}, None),
        ([1, 2, 3, 4], {}, None),
        ([], {1: "E", 2: "E", 3: "E", 4: "E"}, 0),
        ([1, 2, 3, 4], {1: "E", 2: "E", 3: "E", 4: "E"}, 1),
        ([1, 2, 3, 4], {1: "E"}, 1),
        ([1, 2, 3, 4], {2: "E"}, 1 / np.log2(3)),
        ([1, 2, 3, 4], {1: "S", 2: "S", 3: "S", 4: "S"}, 1),
        ([1, 2, 3, 4], {1: "I", 2: "I", 3: "I", 4: "I"}, None),
        ([1, 2, 3, 4], {1: "E", 2: "E", 3: "E", 4: "I"}, 1),
        ([1, 2], {1: "I", 2: "E"}, 1 / np.log2(3)),
    ],
)
def test_compute_ndcg(retrieved_ids, id_to_label, expected):
    actual = compute_ndcg(retrieved_ids, id_to_label, prime=False)
    assert actual == (round(expected, 4) if expected is not None else None)


@pytest.mark.parametrize(
    ("retrieved_ids", "id_to_label", "expected"),
    [
        ([], {}, None),
        ([], {1: "E"}, 0.0),
        ([1, 2, 3, 4], {1: "E"}, 1),
        ([1, 2, 3, 4], {2: "E"}, 1),
        ([1, 2, 3, 4], {4: "E"}, 1),
        ([1, 2], {1: "I", 2: "E"}, 1 / np.log2(3)),
    ],
)
def test_compute_ndcg_prime(retrieved_ids, id_to_label, expected):
    actual = compute_ndcg(retrieved_ids, id_to_label, prime=True)
    assert actual == (round(expected, 4) if expected is not None else None)


def test_compute_cosine_similarity():
    query_vector = np.array([1.0, 1.0])
    product_vectors = np.array([[1.0, 1.0], [-1.0, -1.0]])
    actual = compute_cosine_similarity(query_vector, product_vectors).tolist()
    assert [round(e) for e in actual] == [1, -1]
