import pytest

from amazon_product_search.retrieval.keyword_extractor import KeywordExtractor


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", set()),
        ("Hello World", {"hello world", "hello", "world"}),
    ],
)
def test_apply_yake(s, expected):
    extractor = KeywordExtractor()
    actual = {text for text, score in extractor.apply_yake(s)}
    assert actual == expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", set()),
        ("Hello World", {"world"}),
    ],
)
def test_apply_position_rank(s, expected):
    extractor = KeywordExtractor()
    actual = {text for text, score in extractor.apply_position_rank(s)}
    assert actual == expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", set()),
        ("Hello World", {"world"}),
    ],
)
def test_apply_multipartite_rank(s, expected):
    extractor = KeywordExtractor()
    actual = {text for text, score in extractor.apply_multipartite_rank(s)}
    assert actual == expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", set()),
        ("Hello World", {"hello", "world"}),
    ],
)
def test_apply_keybert(s, expected):
    extractor = KeywordExtractor()
    actual = {text for text, score in extractor.apply_keybert(s)}
    assert actual == expected
