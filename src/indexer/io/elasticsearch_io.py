import json
from typing import Any, Callable, Dict, List, Optional, Union

import apache_beam as beam
from apache_beam import PCollection
from elasticsearch import Elasticsearch, helpers


class WriteToElasticsearch(beam.PTransform):
    """PTransform for writing to Elasticsearch."""

    def __init__(
        self,
        es_host: str,
        index_name: str,
        max_batch_size: int = 10,
        max_batch_size_bytes: int = 10000,
        id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        is_delete_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        build_es_client_fn: Optional[Callable[[str], Elasticsearch]] = None,
    ):
        """Initialize WriteToElasticsearch Transform."""
        self.write_fn = BulkWriteFn(
            es_host=es_host,
            index_name=index_name,
            id_fn=id_fn,
            is_delete_fn=is_delete_fn,
            max_batch_size=max_batch_size,
            max_batch_size_bytes=max_batch_size_bytes,
            build_es_client_fn=build_es_client_fn,
        )

    def expand(self, pcoll: PCollection[Dict[str, Any]]):
        return pcoll | beam.ParDo(self.write_fn)


class BulkWriteFn(beam.DoFn):
    def __init__(
        self,
        es_host: str,
        index_name: str,
        max_batch_size: int,
        max_batch_size_bytes: int,
        id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        is_delete_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        build_es_client_fn: Optional[Callable[[str], Elasticsearch]] = None,
    ):
        self.es_host = es_host
        self.index_name = index_name
        self.id_fn = id_fn
        self.is_delete_fn = is_delete_fn
        self.max_batch_size = max_batch_size
        self.max_batch_size_bytes = max_batch_size_bytes
        self.build_es_client_fn = build_es_client_fn

    def setup(self):
        self.es_client = Elasticsearch(self.es_host)

    def start_bundle(self):
        self.batch = []
        self.current_batch_size_bytes = 0

    def doc_to_action(self, doc: Dict[str, Any]):
        """
        Python helpers.bulk actions:
        [
          {
            '_op_type': 'delete',
            '_index': 'index-name',
            '_id': 42,
          }
          {
            '_op_type': 'update',
            '_index': 'index-name',
            '_id': 42,
            '_source': {'name': 'iphone', ...}
          }
        ]
        """
        is_delete = self.is_delete_fn(doc) if self.is_delete_fn else False
        action: Dict[str, Any] = {
            "_op_type": "delete" if is_delete else "index",
            "_index": self.index_name,
        }
        if self.id_fn:
            action["_id"] = self.id_fn(doc)
        if not is_delete:
            action["_source"] = doc
        return action

    def process(self, doc: Dict[str, Any]):
        action = self.doc_to_action(doc)
        self.batch.append(action)
        self.current_batch_size_bytes += len(bytes(json.dumps(action), "utf-8"))
        if self.current_batch_size_bytes >= self.max_batch_size_bytes or len(self.batch) >= self.max_batch_size:
            self.flush_batch()

    def finish_bundle(self):
        self.flush_batch()

    def flush_batch(self) -> tuple[int, Union[int, List[Dict[str, Any]]]]:
        """Call Bulk API with the request body consisting of multiple actions.

        We can choose whether the Bulk API raises an error (`BulkIndexError`) when one of the action in the batch fails.
        `raise_on_error=False` is set so that the caller is free to handle error responses.
        See: https://elasticsearch-py.readthedocs.io/en/v8.4.2/helpers.html

        Once the request finishes, resets `batch` (list of actions) and `current_batch_size_bytes` are reset.

        Returns:
            tuple[int, Union[int, List[Dict[str, Any]]]]: The first element represents the number of successful actions
                and the second element can be either:
                * The number of error actions (when `stats_only=True`)
                * A list of error responses (when `stats_only=False`)
        """
        if not self.batch:
            return 0, 0

        success, errors = helpers.bulk(
            self.es_client,
            self.batch,
            stats_only=True,
            raise_on_error=False,
        )

        # TODO: Keep track of the number of errors.

        self.batch.clear()
        self.current_batch_size_bytes = 0
        return success, errors

    def teardown(self):
        if self.es_client:
            self.es_client.close()
