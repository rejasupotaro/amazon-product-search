import logging
from typing import Any, Callable, Dict, List

import apache_beam as beam

from amazon_product_search.core.vespa.vespa_client import VespaClient


class WriteToVespa(beam.DoFn):
    def __init__(
        self, host: str, schema: str, id_fn: Callable[[Dict[str, Any]], str]
    ) -> None:
        self.host = host
        self.schema = schema
        self.id_fn = id_fn

    def setup(self) -> None:
        self.client = VespaClient(self.host)

    def process(self, docs: List[Dict[str, Any]]) -> None:
        logging.info(f"Index {len(docs)} docs in a batch")
        for doc in docs:
            for key, value in doc.items():
                if "vector" in key:
                    doc[key] = [float(v) for v in list(value)]
        responses = self.client.feed(self.schema, docs, self.id_fn)
        for response in responses:
            if response.status_code != 200:
                logging.error(response.json)
