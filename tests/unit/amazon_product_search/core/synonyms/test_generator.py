import pytest
import polars as pl

from amazon_product_search.core.synonyms.generator import (
    generate_candidates,
    preprocess_query_title_pairs,
    generate_ngrams,
    generate_ngrams_all,
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


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, ["ab", "cd", "ef", "gh"]),
        (2, ["ab cd", "cd ef", "ef gh"]),
        (3, ["ab cd ef", "cd ef gh"]),
        (4, ["ab cd ef gh"]),
        (5, []),
    ]
)
def test_generate_ngrams(n, expected):
    tokens = ["ab", "cd", "ef", "gh"]
    ngrams = generate_ngrams(tokens, n)
    assert ngrams == expected


def test_generate_ngrams_all():
    tokens = ["ab", "cd", "ef", "gh"]
    ngrams = generate_ngrams_all(tokens, 3)
    assert ngrams == ["ab", "cd", "ef", "gh", "ab cd", "cd ef", "ef gh", "ab cd ef", "cd ef gh"]


def test_generate_candidates():
    pairs = [["a b", "a b"], ["a c", "b c"]]
    candidates = generate_candidates(locale="us", pairs=pairs).to_dicts()
    assert len(candidates) == 4
    assert set(candidates[0].keys()) == {
        "cooccurrence",
        "npmi",
        "query",
        "query_count",
        "title",
        "title_count",
    }
