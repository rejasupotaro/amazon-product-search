from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.response import Response, Result


def test_generate_actions():
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
            "_source": {
                "product_id": "1",
                "product_title": "Product Title",
            },
        },
    ]
    actual = EsClient._generate_actions(
        index_name="products",
        docs=docs,
        id_fn=lambda doc: doc["product_id"],
    )
    actual = list(actual)
    assert actual == expected


def test_convert_es_response_to_response():
    es_response = {
        "took": 3,
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        "hits": {
            "total": {"value": 100, "relation": "eq"},
            "max_score": 1.0,
            "hits": [
                {
                    "_index": "products_jp",
                    "_id": "product_id",
                    "_score": 1.0,
                    "_source": {
                        "product_id": "product_id",
                        "product_title": "title",
                    },
                },
            ],
        },
    }
    expected = Response(
        results=[
            Result(
                product={
                    "product_id": "product_id",
                    "product_title": "title",
                },
                score=1.0,
            )
        ],
        total_hits=100,
    )
    actual = EsClient._convert_es_response_to_response(es_response)
    assert actual == expected
