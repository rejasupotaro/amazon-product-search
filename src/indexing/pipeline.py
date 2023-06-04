import logging
from typing import Any, Dict, Tuple

import apache_beam as beam
import polars as pl
from apache_beam.transforms.ptransform import PTransform
from apache_beam.transforms.util import BatchElements
from apache_beam.utils.shared import Shared

from amazon_product_search import source
from amazon_product_search.source import Locale
from indexing.io.elasticsearch_io import WriteToElasticsearch
from indexing.io.vespa_io import WriteToVespa
from indexing.options import IndexerOptions
from indexing.transforms.analyze_fn import AnalyzeFn
from indexing.transforms.encode_fn import BatchEncodeFn
from indexing.transforms.extract_keywords_fn import (
    ExtractKeywordsFn,
)
from indexing.transforms.filters import is_indexable


def get_input_source(data_dir: str, locale: Locale, nrows: int = -1) -> PTransform:
    products_df = source.load_products(locale, nrows, data_dir)
    products_df = products_df.with_columns(pl.col("*").fill_null(pl.lit("")))
    products = products_df.to_dicts()
    logging.info(f"We have {len(products)} products to index")
    return beam.Create(products)


def join_branches(kv: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    (product_id, group) = kv
    product = group["product"][-1]

    if "extracted_keywords" in group:
        product |= group["extracted_keywords"][-1]

    if "product_vector" in group:
        product["product_vector"] = group["product_vector"][-1]

    return product


def create_pipeline(options: IndexerOptions) -> beam.Pipeline:
    index_name = options.index_name
    locale = options.locale
    data_dir = options.data_dir
    nrows = options.nrows
    text_fields = ["product_title", "product_description", "product_bullet_point"]

    pipeline = beam.Pipeline(options=options)
    products = (
        pipeline
        | get_input_source(data_dir, locale, nrows)
        | "Filter products" >> beam.Filter(is_indexable)
        | "Analyze products" >> beam.ParDo(AnalyzeFn(text_fields))
    )

    branches = {}
    if options.extract_keywords:
        branches["extracted_keywords"] = products | "Extract keywords" >> beam.ParDo(
            ExtractKeywordsFn()
        )
    if options.encode_text:
        branches["product_vector"] = (
            products
            | "Batch products for encoding" >> BatchElements(min_batch_size=8)
            | "Encode products" >> beam.ParDo(BatchEncodeFn(shared_handle=Shared()))
        )
    if branches:
        branches["product"] = products | beam.WithKeys(
            lambda product: product["product_id"]
        )
        products = branches | beam.CoGroupByKey() | beam.Map(join_branches)

    batched_products = products | "Batch products for indexing" >> BatchElements()

    match options.dest:
        case "stdout":
            products | beam.Map(lambda batched_products: logging.info(batched_products))
        case "es":
            (
                batched_products
                | "Index products"
                >> beam.ParDo(
                    WriteToElasticsearch(
                        es_host=options.dest_host,
                        index_name=index_name,
                        id_fn=lambda doc: doc["product_id"],
                    )
                )
            )
        case "vespa":
            (
                batched_products
                | "Index products"
                >> beam.ParDo(
                    WriteToVespa(
                        host=options.dest_host,
                        schema=index_name,
                        id_fn=lambda doc: doc["product_id"],
                    )
                )
            )
    return pipeline


def run(options: IndexerOptions) -> None:
    pipeline = create_pipeline(options)
    result = pipeline.run()
    result.wait_until_finish()
