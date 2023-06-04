import math
from unittest.mock import patch

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from indexing.io.elasticsearch_io import WriteToElasticsearch


@patch("amazon_product_search.es.es_client.Elasticsearch")
@patch("amazon_product_search.es.es_client.EsClient.index_docs", return_value=(0, 0))
def test_bulk_called_n_times(mock_es, mock_es_client):
    products = [
        {
            "product_id": "1",
            "product_title": "Product Title",
        },
    ]

    for batch_size in [1, 2, 3]:
        mock_es_client.reset_mock()
        with TestPipeline() as pipeline:
            (
                pipeline
                | beam.Create(products)
                | beam.BatchElements(
                    min_batch_size=batch_size, max_batch_size=batch_size
                )
                | beam.ParDo(
                    WriteToElasticsearch(
                        es_host="http://localhost:9200",
                        index_name="products",
                    )
                )
            )
        assert mock_es_client.call_count == math.ceil(len(products) / batch_size)
