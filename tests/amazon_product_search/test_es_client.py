from unittest.mock import patch

from amazon_product_search.es_client import EsClient


@patch("amazon_product_search.es_client.Elasticsearch")
def test_doc_to_action(mock_es):
    es_client = EsClient(
        es_host="http://localhost:9200",
    )
    docs = [
        {
            "product_id": "1",
            "product_title": "Product Title",
        }
    ]
    expected = [
        {
            "_op_type": "index",
            "_index": "products",
            "_id": "1",
            "doc": {
                "product_id": "1",
                "product_title": "Product Title",
            },
        },
    ]
    actual = es_client.generate_actions(
        index_name="products",
        docs=docs,
        id_fn=lambda doc: doc["product_id"],
    )
    actual = list(actual)
    assert actual == expected
