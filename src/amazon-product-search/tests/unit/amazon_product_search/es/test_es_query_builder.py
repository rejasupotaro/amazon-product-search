from amazon_product_search.es.query_builder import QueryBuilder
from amazon_product_search.synonyms.synonym_dict import SynonymDict


def test_match_all():
    query_builder = QueryBuilder(locale="us")
    assert query_builder.match_all() == {"match_all": {}}


def test_build_search_query():
    query_builder = QueryBuilder(locale="us")
    es_query = query_builder.build_lexical_search_query(query="query", fields=["product_title"])
    assert es_query == {
        "function_score": {
            "query": {
                "dis_max": {
                    "queries": [
                        {
                            "multi_match": {
                                "query": "query",
                                "type": "cross_fields",
                                "fields": ["product_title"],
                                "operator": "and",
                                "boost": 1.0,
                            },
                        },
                    ],
                },
            },
            "functions": [],
        },
    }


def test_build_search_query_with_synonym_expansion_enabled():
    synonym_dict = SynonymDict(locale="us")
    synonym_dict._entry_dict = {"query": [("synonym", 1.0)]}
    query_builder = QueryBuilder(locale="us", synonym_dict=synonym_dict)
    es_query = query_builder.build_lexical_search_query(
        query="query",
        fields=["product_title"],
        enable_synonym_expansion=True,
    )
    assert es_query == {
        "function_score": {
            "query": {
                "dis_max": {
                    "queries": [
                        {
                            "multi_match": {
                                "query": "query",
                                "type": "cross_fields",
                                "fields": ["product_title"],
                                "operator": "and",
                                "boost": 1.0,
                            },
                        },
                        {
                            "multi_match": {
                                "query": "synonym",
                                "type": "cross_fields",
                                "fields": ["product_title"],
                                "operator": "and",
                                "boost": 0.5,
                            },
                        },
                    ],
                },
            },
            "functions": [],
        },
    }


def test_build_search_query_with_product_ids():
    query_builder = QueryBuilder(locale="us")
    es_query = query_builder.build_lexical_search_query(
        query="query",
        fields=["product_title"],
        product_ids=["1", "2", "3"],
    )
    assert es_query == {
        "bool": {
            "must": [
                {
                    "function_score": {
                        "query": {
                            "dis_max": {
                                "queries": [
                                    {
                                        "multi_match": {
                                            "query": "query",
                                            "type": "cross_fields",
                                            "fields": ["product_title"],
                                            "operator": "and",
                                            "boost": 1.0,
                                        },
                                    },
                                ],
                            },
                        },
                        "functions": [],
                    },
                }
            ],
            "filter": [
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
    es_query = query_builder.build_lexical_search_query(
        query="query",
        fields=["product_title"],
        enable_synonym_expansion=True,
        product_ids=["1", "2", "3"],
    )
    assert es_query == {
        "bool": {
            "must": [
                {
                    "function_score": {
                        "query": {
                            "dis_max": {
                                "queries": [
                                    {
                                        "multi_match": {
                                            "query": "query",
                                            "type": "cross_fields",
                                            "fields": ["product_title"],
                                            "operator": "and",
                                            "boost": 1.0,
                                        },
                                    },
                                    {
                                        "multi_match": {
                                            "query": "synonym",
                                            "type": "cross_fields",
                                            "fields": ["product_title"],
                                            "operator": "and",
                                            "boost": 0.5,
                                        },
                                    },
                                ],
                            },
                        },
                        "functions": [],
                    },
                },
            ],
            "filter": [
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
    es_query = query_builder.build_semantic_search_query(query="query", field="product_vector", top_k=10)
    assert es_query.keys() == {"query_vector", "field", "k", "num_candidates", "boost"}


def test_build_knn_search_query_with_product_id():
    query_builder = QueryBuilder(locale="us")
    es_query = query_builder.build_semantic_search_query(
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
