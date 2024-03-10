from amazon_product_search.core.es.query_builder import QueryBuilder, expand_synonyms
from amazon_product_search.core.synonyms.synonym_dict import SynonymDict


def test_expand_synonyms():
    token_chain = [("1", []), ("2", ["3", "4"]), ("5", ["6"])]
    expanded_queries = []
    expand_synonyms(token_chain, [], expanded_queries)
    assert len(expanded_queries) == 6
    assert expanded_queries == [
        ["1", "2", "5"],
        ["1", "2", "6"],
        ["1", "3", "5"],
        ["1", "3", "6"],
        ["1", "4", "5"],
        ["1", "4", "6"],
    ]


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
        is_synonym_expansion_enabled=True,
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
                                "boost": 1.0,
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
        is_synonym_expansion_enabled=True,
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
                                            "boost": 1.0,
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
