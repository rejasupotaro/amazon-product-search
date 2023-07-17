import pandas as pd
import pytest

from amazon_product_search.training.fine_tuning.components import ProductMLMDataModule


@pytest.mark.parametrize(
    ("prepend_tag", "expected"),
    [
        (False, ["title brand color bullet_point"]),
        (True, ["title: title brand: brand color: color bullet: bullet_point"]),
    ],
)
def test_make_sentences(prepend_tag, expected):
    df = pd.DataFrame(
        [
            {
                "product_title": "title",
                "product_brand": "brand",
                "product_color": "color",
                "product_bullet_point": "bullet_point",
                "product_description": None,
            },
        ]
    )
    actual = ProductMLMDataModule.make_sentences(df, prepend_tag)
    assert actual == expected
