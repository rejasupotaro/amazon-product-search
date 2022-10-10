import math
from unittest.mock import patch

import apache_beam as beam
import pytest
from apache_beam.testing.test_pipeline import TestPipeline

from indexer.io.elasticsearch_io import BulkWriteFn, WriteToElasticsearch

PRODUCTS = [
    {
        "product_id": "1",
        "product_title": "Product Title",
    },
]


@pytest.mark.parametrize(
    "items, expected",
    [
        (
            PRODUCTS,
            [
                {
                    "_op_type": "index",
                    "_index": "items",
                    "_id": "1",
                    "_source": {
                        "product_id": "1",
                        "product_title": "Product Title",
                    },
                },
            ],
        ),
    ],
)
def test_doc_to_action(items, expected):
    write_fn = BulkWriteFn(
        es_host="http://localhost:9200",
        index_name="items",
        max_batch_size=10,
        max_batch_size_bytes=10000,
        id_fn=lambda doc: doc["product_id"],
        is_delete_fn=lambda doc: False,
    )

    for i in range(len(items)):
        actual = write_fn.doc_to_action(items[i])
        assert actual == expected[i]


@patch("indexer.io.elasticsearch_io.helpers.bulk", return_value=(0, 0))
@patch("indexer.io.elasticsearch_io.Elasticsearch")
def test_bulk_called_n_times(mock_es, mock_bulk):
    for batch_size in [1, 2, 3]:
        mock_bulk.reset_mock()
        with TestPipeline() as pipeline:
            (
                pipeline
                | beam.Create(PRODUCTS)
                | WriteToElasticsearch(
                    es_host="http://localhost:9200",
                    index_name="items",
                    max_batch_size=batch_size,
                    max_batch_size_bytes=10000,
                )
            )
        assert mock_bulk.call_count == math.ceil(len(PRODUCTS) / batch_size)
