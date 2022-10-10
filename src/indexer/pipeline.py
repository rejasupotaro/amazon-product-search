import logging

import apache_beam as beam
from apache_beam.transforms.ptransform import PTransform

from amazon_product_search import source


def get_input_source() -> PTransform:
    products_df = source.load_products(locale="jp", nrows=10)
    products = products_df.to_dict("records")
    return beam.Create(products)


def run():
    logging.getLogger().setLevel(logging.INFO)
    input_source = get_input_source()
    with beam.Pipeline() as pipeline:
        (pipeline | input_source | beam.Map(logging.info))
