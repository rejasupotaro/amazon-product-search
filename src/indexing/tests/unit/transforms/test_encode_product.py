import pytest

from indexing.transforms.encode_product import _product_to_text


@pytest.mark.parametrize(
    ("product", "fields", "expected"),
    [
        (
            {"product_title": "product title"},
            ["product_title"],
            "product title",
        ),
        (
            {"product_title": "product title", "product_description": "product description"},
            ["product_title", "product_description"],
            "product title product description",
        ),
    ],
)
def test_product_to_text(product, fields, expected):
    actual = _product_to_text(product, fields)
    assert actual == expected
