import pytest

from amazon_product_search.nlp.normalizer import escape_json, normalize_doc, normalize_query


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("", ""),
        ("\n", ""),
        ('"Hello"', "Hello"),
        ("Joe's Kitchen", "Joe s Kitchen"),
    ],
)
def test_escape_json(s, expected):
    actual = escape_json(s)
    assert actual == expected


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("", ""),
        (
            "商品写真は、撮影条件などのの影響により、実物とは色味に多少の差異がみられる場合が御座います。<br> あらかじめご了承いただきますようお願い申し上げます。<br> <br>",  # noqa: E501
            "商品写真は 撮影条件などのの影響により 実物とは色味に多少の差異がみられる場合が御座います あらかじめご了承いただきますようお願い申し上げます",  # noqa: E501
        ),
    ],
)
def test_normalize_doc(s, expected):
    actual = normalize_doc(s)
    assert actual == expected


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("", ""),
        ("Joe's Kitchen", "joe s kitchen"),
        ("  LOUIS VUITTON", "louis vuitton"),
    ],
)
def test_normalize_query(s, expected):
    actual = normalize_query(s)
    assert actual == expected
