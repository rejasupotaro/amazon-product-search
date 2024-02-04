import pytest

from amazon_product_search.constants import HF
from amazon_product_search.core.retrieval.query_vector_cache import QueryVectorCache
from amazon_product_search.core.synonyms.synonym_dict import SynonymDict
from amazon_product_search.core.vespa.query_builder import QueryBuilder


@pytest.mark.parametrize(
    ("operator", "entry_dict", "expected"),
    [
        (
            "and",
            {},
            "(field1 contains 'token1' OR field2 contains 'token1') AND (field1 contains 'token2' OR field2 contains 'token2')",  # noqa: E501
        ),
        (
            "and",
            {"token1": [("synonym1", 1.0)]},
            "(field1 contains equiv('token1', 'synonym1') OR field2 contains equiv('token1', 'synonym1')) AND (field1 contains 'token2' OR field2 contains 'token2')",  # noqa: E501
        ),
        (
            "weakAnd",
            {"token1": [("synonym1", 1.0)]},
            "userQuery()",
        ),
    ],
)
def test_build_text_matching_query(operator, entry_dict, expected):
    locale = "us"
    synonym_dict = SynonymDict(locale)
    synonym_dict._entry_dict = entry_dict

    query_builder = QueryBuilder(
        locale,
        hf_model_name=HF.LOCALE_TO_MODEL_NAME[locale],
        synonym_dict=synonym_dict,
        vector_cache=QueryVectorCache(),
    )
    query = query_builder._build_text_matching_query(
        tokens=["token1", "token2"],
        fields=["field1", "field2"],
        operator=operator,
    )
    assert query == expected
