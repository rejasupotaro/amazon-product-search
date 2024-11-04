import logging
from typing import Any, Callable, Dict, List

import apache_beam as beam
from vespa.io import VespaResponse

from amazon_product_search.vespa.vespa_client import VespaClient


def callback_fn(response: VespaResponse, id: str) -> None:
    if response.status_code != 200:
        logging.error(f"Failed to index doc: {id}")
        logging.error(response.json)


class WriteToVespa(beam.DoFn):
    def __init__(self, host: str, schema: str, id_fn: Callable[[Dict[str, Any]], str]) -> None:
        self.host = host
        self.schema = schema
        self.id_fn = id_fn

    def setup(self) -> None:
        self.client = VespaClient(self.host)

    def process(self, docs: List[Dict[str, Any]]) -> None:
        logging.info(f"Index {len(docs)} docs in a batch")
        self.client.feed(self.schema, docs, self.id_fn, callback_fn)
