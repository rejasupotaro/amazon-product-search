import logging

import apache_beam as beam
from apache_beam.transforms.ptransform import PTransform

from amazon_product_search import source
from indexer.transforms import AnalyzeFn


def get_input_source() -> PTransform:
    products_df = source.load_products(locale="jp", nrows=10)
    products_df = products_df.fillna("")
    products = products_df.to_dict("records")
    return beam.Create(products)


def run():
    logging.getLogger().setLevel(logging.INFO)
    input_source = get_input_source()
    text_fields = ["product_title", "product_description", "product_bullet_point"]
    with beam.Pipeline() as pipeline:
        (
            pipeline
            | input_source
            | beam.ParDo(AnalyzeFn(text_fields))
            | beam.Map(lambda product: logging.info(product))
            # | beam.Map(lambda product: logging.info(json.dumps(product, indent=4)))
        )
