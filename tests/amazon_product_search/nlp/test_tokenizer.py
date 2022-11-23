import pytest

from amazon_product_search.nlp.tokenizer import DicType, Tokenizer


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", ""),
        ("ナイキ ユニクロ", "ナイキ ユニ クロ"),
        (
            "あらかじめご了承いただきますようお願い申し上げます",  # noqa
            "あらかじめ ご 了承 いただき ます よう お願い 申し上げ ます",  # noqa
        ),
    ],
)
def test_tokenize_with_ipadic(s, expected):
    t = Tokenizer(DicType.IPADIC)
    actual = " ".join(t.tokenize(s))
    assert actual == expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", ""),
        ("ナイキ ユニクロ", "ナイキ ユニ クロ"),
        (
            "あらかじめご了承いただきますようお願い申し上げます",  # noqa
            "あらかじめ ご 了承 いただき ます よう お願い 申し上げ ます",  # noqa
        ),
    ],
)
def test_tokenize_with_unidic(s, expected):
    t = Tokenizer(DicType.IPADIC)
    actual = " ".join(t.tokenize(s))
    assert actual == expected
