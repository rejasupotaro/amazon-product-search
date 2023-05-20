from unittest.mock import patch

from amazon_product_search.es.query_builder import QueryBuilder


def test_build_search_query():
    query_builder = QueryBuilder()
    es_query = query_builder.build_sparse_search_query(
        query="query", fields=["product_title"], query_type="combined_fields"
    )
    assert es_query == {
        "combined_fields": {
            "query": "query",
            "fields": ["product_title"],
            "operator": "and",
            "boost": 1.0,
        }
    }


@patch("amazon_product_search.synonyms.synonym_dict.SynonymDict.load_synonym_dict")
def test_build_search_query_with_synonym_expansion_enabled(mock_method):
    mock_method.return_value = {"query": [("synonym", 1.0), ("antonym", 0.1)]}

    query_builder = QueryBuilder()
    es_query = query_builder.build_sparse_search_query(
        query="query", fields=["product_title"], query_type="combined_fields", is_synonym_expansion_enabled=True
    )
    assert es_query == {
        "bool": {
            "should": [
                {
                    "combined_fields": {
                        "query": "query",
                        "fields": ["product_title"],
                        "operator": "and",
                        "boost": 1.0,
                    },
                },
                {
                    "combined_fields": {
                        "query": "synonym",
                        "fields": ["product_title"],
                        "operator": "and",
                        "boost": 1.0,
                    },
                },
            ],
            "minimum_should_match": 1,
        },
    }


def test_build_knn_search_query():
    query_builder = QueryBuilder()
    es_query = query_builder.build_dense_search_query(query="query", field="product_vector", top_k=10)
    assert es_query.keys() == {"query_vector", "field", "k", "num_candidates", "boost"}
