import logging
from typing import Any, Callable, Dict, List, Optional

import apache_beam as beam
from apache_beam import PCollection
from apache_beam.transforms.util import BatchElements

from amazon_product_search.es.es_client import EsClient


class WriteToElasticsearch(beam.PTransform):
    def __init__(
        self,
        es_host: str,
        index_name: str,
        max_batch_size: int = 100,
        id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        self.max_batch_size = max_batch_size
        self.write_fn = BulkWriteFn(
            es_host=es_host,
            index_name=index_name,
            id_fn=id_fn,
        )

    def expand(self, pcoll: PCollection[Dict[str, Any]]):
        return pcoll | "Batch products for indexing" > BatchElements(
            min_batch_size=8, max_batch_size=self.max_batch_size
        ) | "Index products" >> beam.ParDo(self.write_fn)


class BulkWriteFn(beam.DoFn):
    def __init__(
        self,
        es_host: str,
        index_name: str,
        id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        self.es_host = es_host
        self.index_name = index_name
        self.id_fn = id_fn

    def setup(self):
        self.es_client = EsClient(self.es_host)

    def process(self, docs: List[Dict[str, Any]]):
        logging.info(f"Indexing docs with batch size: {len(docs)}")
        self.es_client.index_docs(self.index_name, docs, id_fn=lambda doc: doc["product_id"])

    def teardown(self):
        if self.es_client:
            self.es_client.close()
