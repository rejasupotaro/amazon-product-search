import logging
import os

import apache_beam as beam
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms.util import BatchElements
from apache_beam.utils.shared import Shared

from amazon_product_search.constants import DATASET_ID, HF, PROJECT_ID
from amazon_product_search.indexing.doc_transformation_pipeline import get_input_source, join_branches
from amazon_product_search.indexing.io.elasticsearch_io import WriteToElasticsearch
from amazon_product_search.indexing.io.vespa_io import WriteToVespa
from amazon_product_search.indexing.options import IndexerOptions
from amazon_product_search.indexing.transforms.analyze_doc_fn import AnalyzeDocFn
from amazon_product_search.indexing.transforms.encode_fn import EncodeInBatchFn
from amazon_product_search.indexing.transforms.extract_keywords_fn import (
    ExtractKeywordsFn,
)
from amazon_product_search.indexing.transforms.filters import is_indexable


def create_pipeline(options: IndexerOptions) -> beam.Pipeline:
    locale = options.locale
    text_fields = [
        "product_title",
        "product_brand",
        "product_color",
        "product_bullet_point",
        "product_description",
    ]
    hf_model_name = HF.LOCALE_TO_MODEL_NAME[locale]

    table_id = f"docs_{locale}"
    project_id = PROJECT_ID if PROJECT_ID else options.view_as(GoogleCloudOptions).project
    table_spec = f"{project_id}:{DATASET_ID}.{table_id}"

    pipeline = beam.Pipeline(options=options)
    match options.source:
        case "file":
            products = (
                pipeline
                | get_input_source(options.data_dir, locale, options.nrows)
                | "Filter products" >> beam.Filter(is_indexable)
                | "Analyze products" >> beam.ParDo(AnalyzeDocFn(text_fields, locale))
            )
            branches = {}
            if options.extract_keywords:
                branches["extracted_keywords"] = products | "Extract keywords" >> beam.ParDo(ExtractKeywordsFn())
            if options.encode_text:
                branches["product_vector"] = (
                    products
                    | "Batch products for encoding" >> BatchElements(min_batch_size=8)
                    | "Encode products"
                    >> beam.ParDo(
                        EncodeInBatchFn(
                            shared_handle=Shared(),
                            hf_model_name=hf_model_name,
                        )
                    )
                )
            if branches:
                branches["product"] = products | beam.WithKeys(lambda product: product["product_id"])
                products = branches | beam.CoGroupByKey() | beam.Map(join_branches)
        case "bq":
            products = pipeline | "Read table" >> beam.io.ReadFromBigQuery(table=table_spec)

    match options.dest:
        case "stdout":
            products | beam.Map(logging.info)
        case "es":
            (
                products
                | "Batch products for WriteToElasticsearch" >> BatchElements()
                | "Index products"
                >> beam.ParDo(
                    WriteToElasticsearch(
                        es_host=options.dest_host,
                        index_name=options.index_name,
                        id_fn=lambda doc: doc["product_id"],
                    )
                )
            )
        case "vespa":
            (
                products
                | "Batch products for WriteToVespa" >> BatchElements()
                | "Index products"
                >> beam.ParDo(
                    WriteToVespa(
                        host=options.dest_host,
                        schema=options.index_name,
                        id_fn=lambda doc: doc["product_id"],
                    )
                )
            )
        case "bq":
            (
                products
                | WriteToBigQuery(
                    table=table_spec,
                    schema=beam.io.SCHEMA_AUTODETECT,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                )
            )
    return pipeline


def run(options: IndexerOptions) -> None:
    pipeline = create_pipeline(options)
    result = pipeline.run()
    result.wait_until_finish()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Disable the warning as suggested:
    # ```
    # huggingface/tokenizers: The current process just got forked,
    # after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    # To disable this warning, you can either:
    # - Avoid using `tokenizers` before the fork if possible
    # - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    # ```
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    options = IndexerOptions()
    run(options)
