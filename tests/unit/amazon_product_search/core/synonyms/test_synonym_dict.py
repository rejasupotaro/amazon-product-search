from amazon_product_search.core.synonyms.synonym_dict import SynonymDict


def test_expand_synonyms():
    for query, expected in [("", []), ("query", ["synonym"])]:
        synonym_dict = SynonymDict(locale="us")
        synonym_dict._entry_dict = {"query": [("synonym", 1.0)]}
        actual = synonym_dict.expand_synonyms(query)
        assert actual == expected
