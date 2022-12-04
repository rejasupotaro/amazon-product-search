from unittest.mock import patch

from amazon_product_search.synonyms.synonym_dict import SynonymDict


@patch("amazon_product_search.synonyms.synonym_dict.SynonymDict.load_synonym_dict")
def test_find_synonyms(mock_method):
    for query, expected in [("", []), ("query", ["synonym"])]:
        mock_method.return_value = {"query": ["synonym"]}
        synonym_dict = SynonymDict()
        actual = synonym_dict.find_synonyms(query)
        assert actual == expected
