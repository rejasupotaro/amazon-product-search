import pytest

from amazon_product_search.core.es.response import Response, Result
from amazon_product_search.core.retrieval.retriever import Retriever, split_fields


def test_split_fields():
    fields = ["title", "description", "vector"]
    sparse_fields, dense_fields = split_fields(fields)
    assert sparse_fields == ["title", "description"]
    assert dense_fields == ["vector"]


def is_sorted(scores: list[float]) -> None:
    i = 0
    try:
        while i < len(scores) - 1:
            assert scores[i] >= scores[i + 1]
            i += 1
    except AssertionError as err:
        raise AssertionError(f"Given scores are not sorted: {scores}") from err


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
            ["1", "2"],
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
            ["1", "2"],
        ),
        (
            [Result(product={"product_id": "1"}, score=0), Result(product={"product_id": "2"}, score=0)],
            [Result(product={"product_id": "3"}, score=0), Result(product={"product_id": "4"}, score=0)],
            4,
            ["3", "1", "4", "2"],
        ),
    ],
)
def test_rrf(sparse_results, dense_results, expected_total_hits, expected_product_ids):
    retriever = Retriever()
    sparse_response = Response(results=sparse_results, total_hits=len(sparse_results))
    dense_response = Response(results=dense_results, total_hits=len(dense_results))
    response = retriever._rrf(sparse_response, dense_response)
    assert response.total_hits == expected_total_hits
    assert [result.product["product_id"] for result in response.results] == expected_product_ids
    is_sorted([result.score for result in response.results])
