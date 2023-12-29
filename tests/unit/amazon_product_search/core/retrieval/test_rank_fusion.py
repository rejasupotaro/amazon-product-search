import numpy as np
import pytest

from amazon_product_search.core.es.response import Response, Result
from amazon_product_search.core.retrieval.rank_fusion import (
    _append_results,
    _borda_counts,
    _merge_responses_by_score,
    _min_max_scores,
    _rrf_scores_with_k,
)


def _is_sorted(scores: list[float]) -> None:
    i = 0
    try:
        while i < len(scores) - 1:
            assert scores[i] >= scores[i + 1]
            i += 1
    except AssertionError as err:
        raise AssertionError(f"Given scores are not sorted: {scores}") from err


@pytest.mark.parametrize(
    ("scores", "expected_scores"),
    [
        ([], []),
        ([1], [1]),
        ([2, 1], [1, 0.5]),
        ([2, 1, 0], [1, 0.5, 0]),
    ],
)
def test_min_max_scores(scores, expected_scores):
    results = [Result(product={"product_id": str(i)}, score=score) for i, score in enumerate(scores)]
    response = Response(results=results, total_hits=len(results))
    response = _min_max_scores(response)
    assert [result.score for result in response.results] == expected_scores


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([], []),
        ([1], [1]),
        ([2, 1], [1, 0.5]),
        ([3, 2, 1], [1, 0.5, 0.3333]),
    ],
)
def test_rrf_scores(scores, expected):
    results = [Result(product={"product_id": str(i)}, score=score) for i, score in enumerate(scores)]
    response = Response(results=results, total_hits=len(results))
    response = _rrf_scores_with_k(response, k=0)
    actual = [result.score for result in response.results]
    assert np.allclose(actual, expected, atol=1e-04)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([], []),
        ([1], [1]),
        ([10, 1], [2, 1]),
        ([100, 10, 1], [3, 2, 1]),
    ],
)
def test_borda_counts(scores, expected):
    results = [Result(product={"product_id": str(i)}, score=score) for i, score in enumerate(scores)]
    response = Response(results=results, total_hits=len(results))
    response = _borda_counts(response, n=len(results))
    actual = [result.score for result in response.results]
    assert np.allclose(actual, expected, atol=1e-04)


@pytest.mark.parametrize(
    ("original_results", "alternative_results", "expected_total_hits", "expected_product_ids"),
    [
        ([], [], 0, []),
        (
            [Result(product={"product_id": "1"}, score=0)],
            [],
            1,
            ["1"],
        ),
        (
            [],
            [Result(product={"product_id": "1"}, score=0)],
            1,
            ["1"],
        ),
        (
            [Result(product={"product_id": "1"}, score=0)],
            [Result(product={"product_id": "2"}, score=0)],
            2,
            ["1", "2"],
        ),
        (
            [
                Result(product={"product_id": "1"}, score=0),
                Result(product={"product_id": "2"}, score=0),
            ],
            [Result(product={"product_id": "3"}, score=0)],
            2,
            ["1", "2"],
        ),
        (
            [Result(product={"product_id": "1"}, score=0)],
            [
                Result(product={"product_id": "2"}, score=0),
                Result(product={"product_id": "3"}, score=0),
            ],
            2,
            ["1", "2"],
        ),
    ],
)
def test_append_results(original_results, alternative_results, expected_total_hits, expected_product_ids):
    sparse_response = Response(results=original_results, total_hits=len(original_results))
    dense_response = Response(results=alternative_results, total_hits=len(alternative_results))
    response = _append_results(sparse_response, dense_response, size=2)
    assert response.total_hits == expected_total_hits
    assert [result.product["product_id"] for result in response.results] == expected_product_ids


@pytest.mark.parametrize(
    ("sparse_results", "dense_results", "expected_total_hits", "expected_product_ids"),
    [
        ([], [], 0, []),
        (
            [Result(product={"product_id": "1"}, score=0)],
            [],
            1,
            ["1"],
        ),
        (
            [Result(product={"product_id": "1"}, score=0)],
            [Result(product={"product_id": "1"}, score=0)],
            1,
            ["1"],
        ),
        (
            [Result(product={"product_id": "1"}, score=0), Result(product={"product_id": "2"}, score=0)],
            [],
            2,
            ["2", "1"],
        ),
        (
            [Result(product={"product_id": "1"}, score=0)],
            [Result(product={"product_id": "2"}, score=0)],
            2,
            ["2", "1"],
        ),
        (
            [Result(product={"product_id": "1"}, score=0), Result(product={"product_id": "2"}, score=0)],
            [Result(product={"product_id": "1"}, score=0)],
            2,
            ["2", "1"],
        ),
        (
            [Result(product={"product_id": "1"}, score=0), Result(product={"product_id": "2"}, score=0)],
            [Result(product={"product_id": "3"}, score=0), Result(product={"product_id": "4"}, score=0)],
            4,
            ["4", "3", "2", "1"],
        ),
    ],
)
def test_merge_responses_by_score(sparse_results, dense_results, expected_total_hits, expected_product_ids):
    sparse_response = Response(results=sparse_results, total_hits=len(sparse_results))
    dense_response = Response(results=dense_results, total_hits=len(dense_results))
    response = _merge_responses_by_score(sparse_response, dense_response, combination_method="sum", size=100)
    assert response.total_hits == expected_total_hits
    assert [result.product["product_id"] for result in response.results] == expected_product_ids
    _is_sorted([result.score for result in response.results])


def test_merge_responses_by_score_with_size():
    sparse_results = [Result(product={"product_id": str(i)}, score=0) for i in range(0, 3)]
    sparse_response = Response(results=sparse_results, total_hits=len(sparse_results))
    dense_results = [Result(product={"product_id": str(i)}, score=0) for i in range(3, 6)]
    dense_response = Response(results=dense_results, total_hits=len(dense_results))
    response = _merge_responses_by_score(sparse_response, dense_response, combination_method="sum", size=4)
    assert response.total_hits == 6
    assert len(response.results) == 4
