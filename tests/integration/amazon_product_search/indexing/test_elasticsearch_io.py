from time import sleep
from typing import Iterator

import apache_beam as beam
import pytest
from apache_beam.transforms.util import BatchElements

from amazon_product_search.core.es.es_client import EsClient
from amazon_product_search.indexing.io.elasticsearch_io import WriteToElasticsearch
from amazon_product_search.indexing.options import IndexerOptions
from tests.integration.amazon_product_search.indexing.es_docker import EsDocker


@pytest.fixture()
def es_docker() -> Iterator[EsDocker]:
    with EsDocker(container_id="amazon_product_search_test") as instance:
        yield instance


def test_pipeline(es_docker):
    inputs = [
        {
            "product_id": "1",
            "product_title": "title",
            "product_description": "description",
        },
    ]
    expected = [
        {
            "product_id": "1",
            "product_title": "title",
            "product_description": "description",
        },
    ]

    locale = "us"
    index_name = "test"
    dest_host = "http://localhost:9200"

    client = EsClient(es_host=dest_host)
    if index_name in client.list_indices():
        client.delete_index(index_name)
    client.create_index(locale, index_name)

    options = IndexerOptions.from_dictionary(
        {
            "index_name": index_name,
            "dest_host": dest_host,
        }
    )
    pipeline = beam.Pipeline(options=options)
    (
        pipeline
        | beam.Create(inputs)
        | "Batch products for WriteToElasticsearch" >> BatchElements()
        | "Index products"
        >> beam.ParDo(
            WriteToElasticsearch(
                es_host=options.dest_host,
                index_name=options.index_name,
                id_fn=lambda doc: doc["product_id"],
            )
        )
    )
    pipeline.run().wait_until_finish()

    # Bulk insert is done in an asynchronous manner. We need to wait for docs to be indexed.
    waited, max_wait = 0, 30
    while waited <= max_wait:
        sleep(1)
        waited += 1
        num_docs = client.count_docs(index_name)
        if num_docs == 0:
            continue
        break

    response = client.search(index_name, query={"match_all": {}})
    actual = [result.product for result in response.results]
    assert actual == expected
