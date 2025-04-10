import logging

import apache_beam as beam
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms.util import BatchElements

from amazon_product_search.constants import DATA_DIR, DATASET_ID, PROJECT_ID
from indexing.io.elasticsearch_io import WriteToElasticsearch
from indexing.io.vespa_io import WriteToVespa
from indexing.options import IndexerOptions
from indexing.pipelines.base import BasePipeline
from indexing.transforms.add_image_url import AddImageUrlFn


class FeedPipeline(BasePipeline):
    def build(self, options: IndexerOptions) -> beam.Pipeline:
        project_id = PROJECT_ID if PROJECT_ID else options.view_as(GoogleCloudOptions).project
        table_spec = f"{project_id}:{DATASET_ID}.{options.table_id}"
        product_images_filepath = f"{DATA_DIR}/product_images.parquet"

        pipeline = beam.Pipeline(options=options)
        products = (
            pipeline
            | "Read table" >> beam.io.ReadFromBigQuery(table=table_spec)
            | "Add image URL" >> beam.ParDo(AddImageUrlFn(filepath=product_images_filepath, locale=options.locale))
        )

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
