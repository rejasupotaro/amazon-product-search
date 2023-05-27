import logging
from typing import Any, Callable, Dict, List, Optional

import apache_beam as beam

from amazon_product_search.es.es_client import EsClient


class WriteToElasticsearch(beam.DoFn):
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
        logging.info(f"Index {len(docs)} docs in a batch")
        self.es_client.index_docs(
            self.index_name, docs, id_fn=lambda doc: doc["product_id"]
        )

    def teardown(self):
        if self.es_client:
            self.es_client.close()
