import pytest

from amazon_product_search.indexer.transforms.filters import is_indexable


@pytest.mark.parametrize(
    ("product", "expected"),
    [
        ({}, False),
        ({"product_title": None}, False),
        ({"product_title": ""}, False),
        ({"product_title": "product_title"}, True),
    ],
)
def test_is_indexable(product, expected):
    actual = is_indexable(product)
    assert actual == expected
