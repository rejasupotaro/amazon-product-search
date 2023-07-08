import pytest

from amazon_product_search.core.retrieval.keyword_extractor import KeywordExtractor


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("", set()),
        ("Hello World", {"hello", "world"}),
    ],
)
def test_apply_keybert(s, expected):
    extractor = KeywordExtractor()
    actual = {text for text, score in extractor.apply_keybert(s)}
    assert actual == expected
