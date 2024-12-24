import apache_beam as beam
import pandas as pd
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from indexing.transforms.add_image_url import AddImageUrlFn


def test_add_image_url(tmp_path):
    filepath = tmp_path / "product_images.parquet"
    pd.DataFrame(
        [
            {
                "asin": "2",
                "locale": "us",
                "image": "https://amazon.product.search/2.jpg",
            }
        ]
    ).to_parquet(filepath, index=False)

    products = [
        {"product_id": "1"},
        {"product_id": "2"},
    ]
    expected = [
        {"product_id": "1"},
        {"product_id": "2", "image_url": "https://amazon.product.search/2.jpg"},
    ]

    with TestPipeline() as pipeline:
        actual = pipeline | beam.Create(products) | beam.ParDo(AddImageUrlFn(filepath=filepath, locale="us"))
        assert_that(actual=actual, matcher=equal_to(expected=expected))
