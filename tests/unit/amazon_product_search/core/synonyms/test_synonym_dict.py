from unittest.mock import patch

from amazon_product_search.core.synonyms.synonym_dict import SynonymDict


@patch("amazon_product_search.core.synonyms.synonym_dict.SynonymDict.load_synonym_dict")
def test_find_synonyms(mock_method):
    mock_method.return_value = {"query": [("synonym", 1.0), ("antonym", 0.1)]}

    for query, expected in [("", []), ("query", ["synonym"])]:
        synonym_dict = SynonymDict()
        actual = synonym_dict.find_synonyms(query)
        assert actual == expected
