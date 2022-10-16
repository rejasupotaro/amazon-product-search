import logging
from typing import Optional

import apache_beam as beam
from apache_beam.transforms.ptransform import PTransform
from sparse_retrieval.indexer.io.elasticsearch_io import WriteToElasticsearch
from sparse_retrieval.indexer.options import IndexerOptions
from sparse_retrieval.indexer.transforms import AnalyzeFn

from amazon_product_search import source


def get_input_source(locale: str, nrows: Optional[int] = None) -> PTransform:
    products_df = source.load_products(locale, nrows)
    products_df = products_df.fillna("")
    products = products_df.to_dict("records")
    logging.info(f"We have {len(products)} products to index")
    return beam.Create(products) | beam.Reshuffle()


def run(options: IndexerOptions):
    locale = options.locale
    es_host = options.es_host
    nrows = options.nrows
    index_name = f"products_{locale}"
    text_fields = ["product_title", "product_description", "product_bullet_point"]

    input_source = get_input_source(locale, nrows)

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
