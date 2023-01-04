from amazon_product_search.es.response import Result
from amazon_product_search.reranking.reranker import ColBERTReranker, DotReranker, NoOpReranker


def test_no_op_reranker():
    product_ids = list(range(10))
    results = [Result(product={"id": product_id}, score=1) for product_id in product_ids]

    reranker = NoOpReranker()
    reranked_results = reranker.rerank("query", results)

    expected = [result.product["id"] for result in results]
    actual = [result.product["id"] for result in reranked_results]
    assert actual == expected


def test_sentence_bert_reranker():
    results = [
        Result(product={"id": "1", "product_title": "xxxxx"}, score=10),
        Result(product={"id": "2", "product_title": "query"}, score=1),
    ]

    reranker = DotReranker()
    reranked_results = reranker.rerank("query", results)

    expected = ["2", "1"]
    actual = [result.product["id"] for result in reranked_results]
    assert actual == expected


def test_colbert_reranker():
    results = [
        Result(product={"id": "1", "product_title": "xxxxx"}, score=10),
        Result(product={"id": "2", "product_title": "query"}, score=1),
    ]

    reranker = ColBERTReranker()
    reranked_results = reranker.rerank("query", results)

    expected = ["2", "1"]
    actual = [result.product["id"] for result in reranked_results]
    assert actual == expected
