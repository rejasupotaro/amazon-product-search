import pytest

from amazon_product_search.core.nlp.analyzer import Analyzer


@pytest.mark.parametrize(
    ("text_fields", "expected"),
    [
        (["field"], {"field": "hello world"}),
        (["unknown_field"], {"field": "HELLO WORLD"}),
    ],
)
def test_analyze(text_fields, expected):
    product = {
        "field": "HELLO WORLD",
    }

    analyzer = Analyzer(text_fields, locale="jp")
    actual = analyzer.analyze(product)
    assert actual == expected
