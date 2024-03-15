import pytest

from amazon_product_search.core.synonyms.synonym_dict import SynonymDict, expand_synonyms


def test_expand_synonyms():
    token_chain = [("1", []), ("2", ["3", "4"]), ("5", ["6"])]
    expanded_queries = []
    expand_synonyms(token_chain, [], expanded_queries)
    assert len(expanded_queries) == 6
    assert expanded_queries == [
        ["1", "2", "5"],
        ["1", "2", "6"],
        ["1", "3", "5"],
        ["1", "3", "6"],
        ["1", "4", "5"],
        ["1", "4", "6"],
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("", []),
        ("a", [("a", ["a'", "a''"])]),
        ("a b", [("a b", ["a b'"])]),
        ("b", [("b", ["b'"])]),
        ("b c", [("b", ["b'"]), ("c", [])]),
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
