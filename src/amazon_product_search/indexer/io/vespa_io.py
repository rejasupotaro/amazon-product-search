import logging
from typing import Any, Callable, Dict, List

import apache_beam as beam

from amazon_product_search.vespa.vespa_client import VespaClient


class WriteToVespa(beam.DoFn):
    def __init__(self, host: str, schema: str, id_fn: Callable[[Dict[str, Any]], str]):
        self.host = host
        self.schema = schema
        self.id_fn = id_fn

    def setup(self):
        self.client = VespaClient(self.host)

    def process(self, docs: List[Dict[str, Any]]):
        logging.info(f"Index {len(docs)} docs in a batch")
        self.client.feed(self.schema, docs, self.id_fn)
