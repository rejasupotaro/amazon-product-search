import polars as pl

from amazon_product_search.synonyms.generator import (
    generate_candidates,
    preprocess_query_title_pairs,
)


def test_preprocess_query_title_pairs():
    df = pl.from_dicts(
        [
            {"query": "Hello", "product_title": "World"},
            {"query": None, "product_title": "product_title"},
            {"query": "query", "product_title": None},
        ]
    )
    df = preprocess_query_title_pairs(df)
    assert df.to_dicts() == [{"query": "hello", "product_title": "world"}]


def test_generate_candidates():
    pairs = [["a b", "a b"], ["a c", "b c"]]
    candidates = generate_candidates(pairs).to_dicts()
    assert len(candidates) == 4
    assert set(candidates[0].keys()) == {
        "cooccurrence",
        "npmi",
        "query",
        "query_count",
        "title",
        "title_count",
    }
