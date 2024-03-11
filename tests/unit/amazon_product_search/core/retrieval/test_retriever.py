from amazon_product_search.core.retrieval.retriever import split_fields


def test_split_fields():
    fields = ["title", "description", "vector"]
    lexical_fields, semantic_fields = split_fields(fields)
    assert lexical_fields == ["title", "description"]
    assert semantic_fields == ["vector"]
