import pytest

from amazon_product_search.core.synonyms.synonym_dict import SynonymDict


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
def test_expand_synonyms(query, expected):
    synonym_dict = SynonymDict(locale="us")
    synonym_dict._entry_dict = {
        "a": [("a'", 1.0), ("a''", 0.5)],
        "b": [("b'", 1.0)],
        "a b": [("a b'", 1.0)],
    }

    actual = synonym_dict.expand_synonyms(query)
    assert actual == expected
