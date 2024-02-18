from amazon_product_search.core.es.query_builder import QueryBuilder
from amazon_product_search.core.synonyms.synonym_dict import SynonymDict


def test_match_all():
    query_builder = QueryBuilder(locale="us")
    assert query_builder.match_all() == {"match_all": {}}


def test_build_search_query():
    query_builder = QueryBuilder(locale="us")
    es_query = query_builder.build_sparse_search_query(
        query="query", fields=["product_title"], query_type="combined_fields"
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
                    }
                }
            ],
            "minimum_should_match": 1,
        }
    }


def test_build_search_query_with_synonym_expansion_enabled():
    synonym_dict = SynonymDict(locale="us")
    synonym_dict._entry_dict = {"query": [("synonym", 1.0)]}
    query_builder = QueryBuilder(locale="us", synonym_dict=synonym_dict)
    es_query = query_builder.build_sparse_search_query(
        query="query",
        fields=["product_title"],
        query_type="combined_fields",
        is_synonym_expansion_enabled=True,
    )
    assert es_query == {
        "bool": {
            "should": [
                {
                    "combined_fields": {
                        "query": "query synonym",
                        "fields": ["product_title"],
                        "operator": "and",
                        "boost": 1.0,
                    },
                },
            ],
            "minimum_should_match": 1,
        },
    }


def test_build_search_query_with_product_ids():
    query_builder = QueryBuilder(locale="us")
    es_query = query_builder.build_sparse_search_query(
        query="query",
        fields=["product_title"],
        query_type="combined_fields",
        product_ids=["1", "2", "3"],
    )
    assert es_query == {
        "bool": {
            "should": [
                {
                    "bool": {
                        "should": [
                            {
                                "combined_fields": {
                                    "query": "query",
                                    "fields": ["product_title"],
                                    "operator": "and",
                                    "boost": 1.0,
                                },
                            }
                        ],
                        "minimum_should_match": 1,
                    },
                },
            ],
            "must": [
                {
                    "terms": {
                        "product_id": ["1", "2", "3"],
                    },
                },
            ],
        }
    }


def test_build_search_query_with_synonym_expansion_enabled_with_product_ids():
    synonym_dict = SynonymDict(locale="us")
    synonym_dict._entry_dict = {"query": [("synonym", 1.0)]}
    query_builder = QueryBuilder(locale="us", synonym_dict=synonym_dict)
    es_query = query_builder.build_sparse_search_query(
        query="query",
        fields=["product_title"],
        query_type="combined_fields",
        is_synonym_expansion_enabled=True,
        product_ids=["1", "2", "3"],
    )
    assert es_query == {
        "bool": {
            "should": [
                {
                    "bool": {
                        "should": [
                            {
                                "combined_fields": {
                                    "query": "query synonym",
                                    "fields": ["product_title"],
                                    "operator": "and",
                                    "boost": 1.0,
                                },
                            },
                        ],
                        "minimum_should_match": 1,
                    },
                },
            ],
            "must": [
                {
                    "terms": {
                        "product_id": ["1", "2", "3"],
                    },
                },
            ],
        },
    }


def test_build_knn_search_query():
    query_builder = QueryBuilder(locale="us")
    es_query = query_builder.build_dense_search_query(query="query", field="product_vector", top_k=10)
    assert es_query.keys() == {"query_vector", "field", "k", "num_candidates", "boost"}


def test_build_knn_search_query_with_product_id():
    query_builder = QueryBuilder(locale="us")
    es_query = query_builder.build_dense_search_query(
        query="query", field="product_vector", top_k=10, product_ids=["1", "2", "3"]
    )
    assert es_query.keys() == {
        "query_vector",
        "field",
        "k",
        "num_candidates",
        "boost",
        "filter",
    }
