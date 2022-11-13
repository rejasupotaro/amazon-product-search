import logging
from typing import Any, Dict, Tuple

import apache_beam as beam
from apache_beam.transforms.ptransform import PTransform
from apache_beam.transforms.util import BatchElements
from apache_beam.utils.shared import Shared

from amazon_product_search import source
from amazon_product_search.indexer.io.elasticsearch_io import WriteToElasticsearch
from amazon_product_search.indexer.options import IndexerOptions
from amazon_product_search.indexer.transforms.analyze_fn import AnalyzeFn
from amazon_product_search.indexer.transforms.encode_fn import BatchEncodeFn
from amazon_product_search.indexer.transforms.extract_keywords_fn import ExtractKeywordsFn
from amazon_product_search.indexer.transforms.filters import is_indexable


def get_input_source(locale: str, nrows: int = -1) -> PTransform:
    products_df = source.load_products(locale, nrows)
    products_df = products_df.fillna("")
    products = products_df.to_dict("records")
    logging.info(f"We have {len(products)} products to index")
    return beam.Create(products)


def join_branches(kv: Tuple[str, Dict[str, Any]]):
    (product_id, group) = kv
    product = group["product"][0]

    if "extracted_keywords" in group:
        product |= group["extracted_keywords"][0]

    if "product_vector" in group:
        product["product_vector"] = group["product_vector"][0]

    return product


def run(options: IndexerOptions):
    index_name = options.index_name
    locale = options.locale
    es_host = options.es_host
    nrows = options.nrows
    text_fields = ["product_title", "product_description", "product_bullet_point"]

    with beam.Pipeline(options=options) as pipeline:
        extractor_shared_handle = Shared()
        encoder_shared_handle = Shared()

        products = (
            pipeline
            | get_input_source(locale, nrows)
            | "Filter products" >> beam.Filter(is_indexable)
            | "Analyze products" >> beam.ParDo(AnalyzeFn(text_fields))
        )

        branches = {
            "product": products | beam.WithKeys(lambda product: product["product_id"]),
        }

        if options.extract_keywords:
            branches["extracted_keywords"] = products | "Extract keywords" >> beam.ParDo(
                ExtractKeywordsFn(shared_handle=extractor_shared_handle)
            )

        if options.encode_text:
            branches["product_vector"] = (
                products
                | "Batch products for encoding" >> BatchElements(min_batch_size=8)
                | "Encode products" >> beam.ParDo(BatchEncodeFn(shared_handle=encoder_shared_handle))
            )

        products = branches | beam.CoGroupByKey() | beam.Map(join_branches)

        if es_host:
            (
                products
                | "Batch products for indexing" >> BatchElements()
                | "Index products"
                >> beam.ParDo(
                    WriteToElasticsearch(
                        es_host=es_host,
                        index_name=index_name,
                        id_fn=lambda doc: doc["product_id"],
                    )
                )
            )
        else:
            products | beam.Map(lambda product: logging.info(product))
