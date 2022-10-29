import pytest

from amazon_product_search.es.query_builder import build_knn_search_query, build_multimatch_search_query


@pytest.mark.parametrize(
    "use_description,expected_fields",
    [
        (False, ["product_title"]),
        (True, ["product_title", "product_description"]),
    ],
)
def test_build_search_query(use_description, expected_fields):
    es_query = build_multimatch_search_query(query="query", use_description=use_description)
    assert es_query["multi_match"]["fields"] == expected_fields


def test_build_knn_search_query():
    es_query = build_knn_search_query(query_vector=[0.1, 0.9], top_k=10)
    assert es_query == {
        "field": "product_vector",
        "k": 10,
        "num_candidates": 100,
        "query_vector": [0.1, 0.9],
    }
