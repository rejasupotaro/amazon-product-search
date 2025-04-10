import logging
from functools import partial
from typing import Any, Dict, Iterator, List

import apache_beam as beam
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms.ptransform import PTransform
from apache_beam.transforms.util import BatchElements
from apache_beam.utils.shared import Shared
from data_source import Locale, loader
from torch import Tensor

from amazon_product_search.constants import DATASET_ID, HF, PROJECT_ID
from amazon_product_search.nlp.normalizer import normalize_query
from dense_retrieval.encoders import SBERTEncoder
from dense_retrieval.encoders.modules.pooler import PoolingMode
from indexing.options import IndexerOptions
from indexing.pipelines.base import BasePipeline


def get_input_source(data_dir: str, locale: Locale, nrows: int = -1) -> PTransform:
    df = loader.load_examples(data_dir, locale, nrows)
    queries = df["query"].unique()
    query_dicts = [{"query": query} for query in queries]
    return beam.Create(query_dicts)


def initialize_encoder(hf_model_name: str, pooling_mode: PoolingMode) -> SBERTEncoder:
    return SBERTEncoder(hf_model_name, pooling_mode)


class NormalizeQueryFn(beam.DoFn):
    def process(self, query_dict: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        query_dict["query"] = normalize_query(query_dict["query"])
        yield query_dict


class EncodeQueriesInBatchFn(beam.DoFn):
    def __init__(self, shared_handle: Shared, hf_model_name: str, pooling_mode: PoolingMode = "mean") -> None:
        self._shared_handle = shared_handle
        self._initialize_fn = partial(initialize_encoder, hf_model_name, pooling_mode)

    def setup(self) -> None:
        self._encoder: SBERTEncoder = self._shared_handle.acquire(self._initialize_fn)

    def _encode(self, texts: list[str]) -> Tensor:
        return self._encoder.encode(texts)

    def process(self, query_dicts: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        logging.info(f"Encode {len(query_dicts)} queries in a batch")
        query_vectors = self._encode([query_dict["query"] for query_dict in query_dicts])
        for query_dict, query_vector in zip(query_dicts, query_vectors, strict=True):
            query_dict["query_vector"] = query_vector.tolist()
            yield query_dict


class QueryPipeline(BasePipeline):
    def build(self, options: IndexerOptions) -> beam.Pipeline:
        locale = options.locale
        hf_model_name = HF.LOCALE_TO_MODEL_NAME[locale]

        project_id = PROJECT_ID if PROJECT_ID else options.view_as(GoogleCloudOptions).project
        table_spec = f"{project_id}:{DATASET_ID}.{options.table_id}"

        pipeline = beam.Pipeline(options=options)
        queries = (
            pipeline
            | get_input_source(options.data_dir, locale, options.nrows)
            | "Analyze queries" >> beam.ParDo(NormalizeQueryFn())
            | "Filter queries" >> beam.Filter(lambda query_dict: bool(query_dict["query"]))
            | "Batch queries for encoding" >> BatchElements(min_batch_size=8)
            | "Encode queries"
            >> beam.ParDo(
                EncodeQueriesInBatchFn(
                    shared_handle=Shared(),
                    hf_model_name=hf_model_name,
                )
            )
        )

        match options.dest:
            case "stdout":
                queries | beam.Map(logging.info)
            case "bq":
                (
                    queries
                    | WriteToBigQuery(
                        table=table_spec,
                        schema=beam.io.SCHEMA_AUTODETECT,
                        write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    )
                )
        return pipeline
