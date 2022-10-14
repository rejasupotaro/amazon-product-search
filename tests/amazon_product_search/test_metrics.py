import numpy as np
import pytest

from amazon_product_search.metrics import compute_ap, compute_ndcg, compute_recall, compute_zero_hit_rate


@pytest.mark.parametrize(
    "xs,expected",
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
    "retrieved_ids,relevant_ids,expected",
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
    "retrieved_ids,relevant_ids,expected",
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
    "retrieved_ids,relevant_ids,expected",
    [
        ([], {}, None),
        ([1, 2, 3, 4], {}, None),
        ([], {1: "exact", 2: "exact", 3: "exact", 4: "exact"}, None),
        ([1, 2, 3, 4], {1: "exact", 2: "exact", 3: "exact", 4: "exact"}, 1),
        ([1, 2, 3, 4], {1: "exact"}, 1),
        ([1, 2, 3, 4], {1: "substitute", 2: "substitute", 3: "substitute", 4: "substitute"}, 1),
        ([1, 2, 3, 4], {1: "irrelevant", 2: "irrelevant", 3: "irrelevant", 4: "irrelevant"}, None),
        ([1, 2, 3, 4], {1: "exact", 2: "exact", 3: "exact", 4: "irrelevant"}, 1),
        ([1, 2], {1: "irrelevant", 2: "exact"}, 1 / np.log2(3)),
    ],
)
def test_compute_ndcg(retrieved_ids, relevant_ids, expected):
    actual = compute_ndcg(retrieved_ids, relevant_ids)
    assert actual == expected
