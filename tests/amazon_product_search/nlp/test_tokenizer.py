import pytest

from amazon_product_search.nlp.tokenizer import Tokenizer, TokenizerType


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", ""),
        (" ", ""),
        ("ナイキ ユニクロ", "ナイキ ユニクロ"),
        (
            "あらかじめご了承いただきますようお願い申し上げます",
            "あらかじめ ご 了承 いただき ます よう お 願い 申し上げ ます",
        ),
    ],
)
def test_tokenize_with_unidic(s, expected):
    t = Tokenizer(TokenizerType.UNIDIC)
    actual = " ".join(t.tokenize(s))
    assert actual == expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", ""),
        (" ", ""),
        ("ナイキ ユニクロ", "ナイキ ユニ クロ"),
        (
            "あらかじめご了承いただきますようお願い申し上げます",
            "あらかじめ ご 了承 いただき ます よう お願い 申し上げ ます",
        ),
    ],
)
def test_tokenize_with_ipadic(s, expected):
    t = Tokenizer(TokenizerType.IPADIC)
    actual = " ".join(t.tokenize(s))
    assert actual == expected
