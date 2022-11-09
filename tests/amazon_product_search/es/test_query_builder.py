from amazon_product_search.es.query_builder import QueryBuilder


def test_build_search_query():
    query_builder = QueryBuilder()
    es_query = query_builder.build_multimatch_search_query(query="query", fields=["product_title"])
    assert es_query == {
        "multi_match": {
            "query": "query",
            "fields": ["product_title"],
            "operator": "or",
        }
    }


def test_build_knn_search_query():
    query_builder = QueryBuilder()
    es_query = query_builder.build_knn_search_query(query="query", field="product_vector", top_k=10)
    assert es_query.keys() == {"query_vector", "field", "k", "num_candidates"}
