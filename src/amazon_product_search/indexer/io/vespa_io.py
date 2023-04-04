import logging
from typing import Any, Dict, List

import apache_beam as beam

from amazon_product_search.vespa.vespa_client import VespaClient


class WriteToVespa(beam.DoFn):
    def __init__(self, host: str, schema: str):
        self.host = host
        self.schema = schema

    def setup(self):
        self.client = VespaClient(self.host)

    def process(self, docs: List[Dict[str, Any]]):
        logging.info(f"Index {len(docs)} docs in a batch")
        batch = [{"id": doc["product_id"], "fields": doc} for doc in docs]
        self.client.feed(self.schema, batch)
