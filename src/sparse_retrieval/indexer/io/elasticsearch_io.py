import logging
from typing import Any, Callable, Dict, List, Optional, Union

import apache_beam as beam
from apache_beam import PCollection

from amazon_product_search.es_client import EsClient


class WriteToElasticsearch(beam.PTransform):
    def __init__(
        self,
        es_host: str,
        index_name: str,
        max_batch_size: int = 100,
        id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        self.write_fn = BulkWriteFn(
            es_host=es_host,
            index_name=index_name,
            max_batch_size=max_batch_size,
            id_fn=id_fn,
        )

    def expand(self, pcoll: PCollection[Dict[str, Any]]):
        return pcoll | beam.ParDo(self.write_fn)


class BulkWriteFn(beam.DoFn):
    def __init__(
        self,
        es_host: str,
        index_name: str,
        max_batch_size: int,
        id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        self.es_host = es_host
        self.index_name = index_name
        self.max_batch_size = max_batch_size
        self.id_fn = id_fn

    def setup(self):
        self.es_client = EsClient(self.es_host)

    def start_bundle(self):
        self.batch: List[Dict[str, Any]] = []

    def process(self, doc: Dict[str, Any]):
        self.batch.append(doc)
        if len(self.batch) >= self.max_batch_size:
            self.flush_batch()

    def finish_bundle(self):
        self.flush_batch()

    def flush_batch(self) -> tuple[int, Union[int, List[Dict[str, Any]]]]:
        """Call Bulk API with the request body consisting of multiple actions.

        We can choose whether the Bulk API raises an error (`BulkIndexError`) when one of the action in the batch fails.
        `raise_on_error=False` is set so that the caller is free to handle error responses.
        See: https://elasticsearch-py.readthedocs.io/en/v8.4.2/helpers.html

        Once the request finishes, resets `batch` (list of actions) is reset.

        Returns:
            tuple[int, Union[int, List[Dict[str, Any]]]]: The first element represents the number of successful actions
                and the second element can be either:
                * The number of error actions (when `stats_only=True`)
                * A list of error responses (when `stats_only=False`)
        """
        if not self.batch:
            return 0, 0
        success, errors = self.es_client.index_docs(self.index_name, self.batch, self.id_fn)
        logging.info(f"Bulk API is called. success: {success}, errors: {errors}")

        self.batch.clear()
        return success, errors

    def teardown(self):
        if self.es_client:
            self.es_client.close()
