import logging
import os
from typing import Any, Dict, Iterator, List

import apache_beam as beam
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms.ptransform import PTransform
from apache_beam.transforms.util import BatchElements
from apache_beam.utils.shared import Shared
from torch import Tensor

from amazon_product_search.constants import HF, PROJECT_ID
from amazon_product_search.core import source
from amazon_product_search.core.source import Locale
from amazon_product_search.indexing.options import IndexerOptions
from amazon_product_search.indexing.transforms.weak_reference import WeakReference
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder
from amazon_product_search_dense_retrieval.encoders.modules.pooler import PoolingMode


class EncodeQueriesInBatchFn(beam.DoFn):
    def __init__(self, shared_handle: Shared, hf_model_name: str, pooling_mode: PoolingMode) -> None:
        self._shared_handle = shared_handle
        self._hf_model_name = hf_model_name
        self._pooling_mode = pooling_mode

    def setup(self) -> None:
        def initialize_encoder() -> WeakReference[SBERTEncoder]:
            # Load a potentially large model in memory. Executed once per process.
            return WeakReference[SBERTEncoder](SBERTEncoder(self._hf_model_name, self._pooling_mode))

        self._weak_reference: WeakReference[SBERTEncoder] = self._shared_handle.acquire(initialize_encoder)

    def _encode(self, texts: list[str]) -> Tensor:
        return self._weak_reference.ref.encode(texts)

    def process(self, query_dicts: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        logging.info(f"Encode {len(query_dicts)} products in a batch")
        query_vectors = self._encode([query_dict["query"] for query_dict in query_dicts])
        for query_dict, query_vector in zip(query_dicts, query_vectors, strict=True):
            query_dict["query_vector"] = [float(v) for v in list(query_vector)]
            yield query_dict


def get_input_source(data_dir: str, locale: Locale, nrows: int = -1) -> PTransform:
    df = source.load_labels(locale, nrows, data_dir)
    queries = df.get_column("query").sample(frac=1).unique()
    query_dicts = {"query": query for query in queries}
    logging.info(f"{len(query_dicts)} queries are going to be encoded")
    return beam.Create(query_dicts)


def create_pipeline(options: IndexerOptions) -> beam.Pipeline:
    locale = options.locale
    hf_model_name, pooling_mode = {
        "us": (HF.EN_MULTIQA, "cls"),
        "jp": (HF.JP_SLUKE_MEAN, "mean"),
    }[locale]

    dataset_id = "amazon"
    table_id = f"queries_{locale}"
    project_id = PROJECT_ID if PROJECT_ID else options.view_as(GoogleCloudOptions).project
    table_spec = f"{project_id}:{dataset_id}.{table_id}"

    pipeline = beam.Pipeline(options=options)
    products = (
        pipeline
        | get_input_source(options.data_dir, locale, options.nrows)
        | "Filter products" >> beam.Filter(lambda query_dict: bool(query_dict["query"]))
        | "Batch products for encoding" >> BatchElements(min_batch_size=8)
        | "Encode products"
        >> beam.ParDo(
            EncodeQueriesInBatchFn(
                shared_handle=Shared(),
                hf_model_name=hf_model_name,
                pooling_mode=pooling_mode,
            )
        )
    )

    match options.dest:
        case "stdout":
            products | beam.Map(logging.info)
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
