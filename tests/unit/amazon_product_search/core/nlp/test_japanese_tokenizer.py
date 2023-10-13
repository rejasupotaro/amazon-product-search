import pytest

from amazon_product_search.core.nlp.tokenizers.japanese_tokenizer import DicType, JapaneseTokenizer, OutputFormat


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("", []),
        (" ", []),
        ("ナイキ ユニクロ", ["ナイキ", "ユニクロ"]),
        (
            "あらかじめご了承いただきますようお願い申し上げます",
            ["あらかじめ", "ご", "了承", "いただき", "ます", "よう", "お", "願い", "申し上げ", "ます"],
        ),
    ],
)
def test_tokenize_with_unidic(s, expected):
    t = JapaneseTokenizer(DicType.UNIDIC)
    actual = t.tokenize(s)
    assert actual == expected


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("", []),
        (" ", []),
        ("ナイキ ユニクロ", ["ナイキ", "ユニ", "クロ"]),
        (
            "あらかじめご了承いただきますようお願い申し上げます",
            ["あらかじめ", "ご", "了承", "いただき", "ます", "よう", "お願い", "申し上げ", "ます"],
        ),
    ],
)
def test_tokenize_with_ipadic(s, expected):
    t = JapaneseTokenizer(DicType.IPADIC)
    actual = t.tokenize(s)
    assert actual == expected


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("", []),
        (
            "キャンプ用品",
            [
                ("キャンプ", ["名詞", "普通名詞", "サ変可能", "*"]),
                ("用", ["接尾辞", "名詞的", "一般", "*"]),
                ("品", ["接尾辞", "名詞的", "一般", "*"]),
            ],
        ),
    ],
)
def test_analyze_with_unidic(s, expected):
    t = JapaneseTokenizer(DicType.UNIDIC, output_format=OutputFormat.DUMP)
    actual = t.tokenize(s)
    assert actual == expected
