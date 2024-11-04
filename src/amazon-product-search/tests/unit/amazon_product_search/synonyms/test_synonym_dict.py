import pytest

from amazon_product_search.synonyms.synonym_dict import SynonymDict, expand_synonyms


def test_expand_synonyms():
    token_chain = [(("a", None), []), (("b", None), [("b'", 1.0), ("b''", 0.5)]), (("c", None), [("c'", 1.0)])]
    expanded_tokens = []
    expand_synonyms(token_chain, [], expanded_tokens)
    assert len(expanded_tokens) == 6
    assert [" ".join([token for token, _ in token_scores]) for token_scores in expanded_tokens] == [
        "a b c",
        "a b c'",
        "a b' c",
        "a b' c'",
        "a b'' c",
        "a b'' c'",
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("", []),
        ("a", [("a", [("a'", 1.0), ("a''", 0.5)])]),
        ("a b", [("a b", [("a b'", 1.0)])]),
        ("b", [("b", [("b'", 1.0)])]),
        ("b c", [("b", [("b'", 1.0)]), ("c", [])]),
        ("c", [("c", [])]),
        ("c d", [("c", []), ("d", [])]),
    ],
)
def test_look_up(query, expected):
    synonym_dict = SynonymDict(locale="us")
    synonym_dict._entry_dict = {
        "a": [("a'", 1.0), ("a''", 0.5)],
        "b": [("b'", 1.0)],
        "a b": [("a b'", 1.0)],
    }

    actual = synonym_dict.look_up(query.split())
    assert actual == expected
