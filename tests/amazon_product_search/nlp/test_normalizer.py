import pytest

from amazon_product_search.nlp.normalizer import normalize


@pytest.mark.parametrize(
    "s,expected",
    [
        (
            "",
            "",
        ),
        (
            "商品写真は、撮影条件などのの影響により、実物とは色味に多少の差異がみられる場合が御座います。<br> あらかじめご了承いただきますようお願い申し上げます。<br> <br>",  # noqa
            "商品写真は 撮影条件などのの影響により 実物とは色味に多少の差異がみられる場合が御座います あらかじめご了承いただきますようお願い申し上げます",
        ),
    ],
)
def test_normalize(s, expected):
    actual = normalize(s)
    assert actual == expected
