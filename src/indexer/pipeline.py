import logging

import apache_beam as beam
from apache_beam.transforms.ptransform import PTransform

from amazon_product_search import source
from indexer.io.elasticsearch_io import WriteToElasticsearch
from indexer.transforms import AnalyzeFn


def get_input_source(locale: str) -> PTransform:
    products_df = source.load_products(locale=locale, nrows=10)
    products_df = products_df.fillna("")
    products = products_df.to_dict("records")
    return beam.Create(products)


def run():
    logging.getLogger().setLevel(logging.INFO)

    locale = "jp"
    text_fields = ["product_title", "product_description", "product_bullet_point"]
    es_host = "http://localhost:9200"
    index_name = f"products_{locale}"

    input_source = get_input_source(locale)

    with beam.Pipeline() as pipeline:
        products = pipeline | input_source | beam.ParDo(AnalyzeFn(text_fields))
        if es_host:
            (
                products
                | WriteToElasticsearch(
                    es_host=es_host,
                    index_name=index_name,
                    id_fn=lambda doc: doc["product_id"],
                )
            )
        else:
            products | beam.Map(lambda product: logging.info(product))
