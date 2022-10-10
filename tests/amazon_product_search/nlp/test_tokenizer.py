import pytest

from amazon_product_search.nlp.tokenizer import Tokenizer


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", ""),
        (
            "あらかじめご了承いただきますようお願い申し上げます",  # noqa
            "あらかじめ ご 了承 いただき ます よう お願い 申し上げ ます",  # noqa
        ),
    ],
)
def test_tokenize(s, expected):
    t = Tokenizer()
    actual = " ".join(t.tokenize(s))
    assert actual == expected
