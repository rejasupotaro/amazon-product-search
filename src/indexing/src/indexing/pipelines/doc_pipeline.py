import logging
from typing import Any, Dict, Tuple

import apache_beam as beam
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms.ptransform import PTransform
from apache_beam.transforms.util import BatchElements
from apache_beam.utils.shared import Shared
from data_source import loader

from amazon_product_search.constants import DATA_DIR, DATASET_ID, HF, PROJECT_ID
from amazon_product_search.source import Locale
from indexing.io.elasticsearch_io import WriteToElasticsearch
from indexing.io.vespa_io import WriteToVespa
from indexing.options import IndexerOptions
from indexing.pipelines.base import BasePipeline
from indexing.transforms.add_image_url import AddImageUrlFn
from indexing.transforms.analyze_doc import AnalyzeDocFn
from indexing.transforms.encode_product import EncodeProduct
from indexing.transforms.filters import is_indexable


def get_input_source(data_dir: str, locale: Locale, nrows: int = -1) -> PTransform:
    products_df = loader.load_products(data_dir, locale)
    if nrows:
        products_df = products_df[:nrows]
    products_df = products_df.fillna("")
    products = products_df.to_dict("records")
    logging.info(f"{len(products)} products are going to be indexed")
    return beam.Create(products)


def join_branches(kv: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    (product_id, group) = kv
    product = group["product"][-1]

    if "product_vector" in group:
        product["product_vector"] = group["product_vector"][-1]

    return product


class DocPipeline(BasePipeline):
    def build(self, options: IndexerOptions) -> beam.Pipeline:
        locale = options.locale
        text_fields = [
            "product_title",
            "product_brand",
            "product_color",
            "product_bullet_point",
            "product_description",
        ]
        hf_model_name = HF.LOCALE_TO_MODEL_NAME[locale]
        product_images_filepath = f"{DATA_DIR}/product_images.parquet"

        pipeline = beam.Pipeline(options=options)
        products = (
            pipeline
            | get_input_source(options.data_dir, locale, options.nrows)
            | "Filter products" >> beam.Filter(is_indexable)
            | "Analyze products" >> beam.ParDo(AnalyzeDocFn(text_fields, locale))
            | "Add image URL" >> beam.ParDo(AddImageUrlFn(product_images_filepath, locale))
        )
        branches = {}
        if options.encode_text:
            branches["product_vector"] = products | "Encode products" >> EncodeProduct(
                Shared(),
                hf_model_name,
                batch_size=8,
            )
        if branches:
            branches["product"] = products | beam.WithKeys(lambda product: product["product_id"])
            products = branches | beam.CoGroupByKey() | beam.Map(join_branches)

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
                project_id = PROJECT_ID if PROJECT_ID else options.view_as(GoogleCloudOptions).project
                table_spec = f"{project_id}:{DATASET_ID}.{options.table_id}"
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
