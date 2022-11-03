from amazon_product_search.es.query_builder import build_knn_search_query, build_multimatch_search_query


def test_build_search_query():
    es_query = build_multimatch_search_query(query="query", fields=["product_title"])
    assert es_query == {
        "multi_match": {
            "query": "query",
            "fields": ["product_title"],
            "operator": "or",
        }
    }


def test_build_knn_search_query():
    es_query = build_knn_search_query(query_vector=[0.1, 0.9], field="product_vector", top_k=10)
    assert es_query == {
        "query_vector": [0.1, 0.9],
        "field": "product_vector",
        "k": 10,
        "num_candidates": 100,
    }
