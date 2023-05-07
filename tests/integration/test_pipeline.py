from pathlib import Path
from time import sleep
from typing import Iterator

import polars as pl
import pytest

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.indexer.options import IndexerOptions
from amazon_product_search.indexer.pipeline import create_pipeline
from tests.integration.es_docker import EsDocker


@pytest.fixture()
def es_docker() -> Iterator[EsDocker]:
    with EsDocker(container_id="amazon_product_search_test") as instance:
        yield instance


def test_pipeline(tmp_path: Path, es_docker):
    inputs = [
        {
            "product_id": "1",
            "product_title": "Title",
            "product_description": "Description",
            "product_bullet_point": "Bullet Point",
            "product_brand": "Brand",
            "product_color": "Color",
        },
    ]
    expected = [
        {
            "product_id": "1",
            "product_title": "title",
            "product_description": "description",
            "product_bullet_point": "bullet point",
            "product_brand": "Brand",
            "product_color": "Color",
            "product_description_keybert": "description point bullet",
        },
    ]

    index_name = "test"
    locale = "jp"
    dest_host = "http://localhost:9200"

    client = EsClient(es_host=dest_host)
    if index_name in client.list_indices():
        client.delete_index(index_name)
    client.create_index(index_name)

    data_dir = tmp_path / "amazon_product_search_test"
    data_dir.mkdir()
    filepath = data_dir / f"products_{locale}.parquet"
    pl.from_dicts(inputs).write_parquet(filepath)

    options = IndexerOptions.from_dictionary(
        {
            "index_name": index_name,
            "locale": locale,
            "data_dir": str(data_dir),
            "dest": "es",
            "dest_host": dest_host,
            "extract_keywords": True,
            "encode_text": True,
        }
    )
    pipeline = create_pipeline(options)
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
