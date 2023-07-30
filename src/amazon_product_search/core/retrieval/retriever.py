from typing import Any

from amazon_product_search.core.es.es_client import EsClient
from amazon_product_search.core.es.query_builder import QueryBuilder
from amazon_product_search.core.es.response import Response, Result
from amazon_product_search.core.nlp.normalizer import normalize_query


def split_fields(fields: list[str]) -> tuple[list[str], list[str]]:
    """Convert a given list of fields into a tuple of (sparse_fields, dense_fields)

    Field names containing "vector" will be considered dense_fields.

    Args:
        fields (list[str]): A list of fields.

    Returns:
        tuple[list[str], list[str]]: A tuple of (sparse_fields, dense_fields)
    """
    sparse_fields: list[str] = []
    dense_fields: list[str] = []
    for field in fields:
        (dense_fields if "vector" in field else sparse_fields).append(field)
    return sparse_fields, dense_fields


class Retriever:
    def __init__(self, es_client: EsClient | None = None, query_builder: QueryBuilder | None = None) -> None:
        if es_client:
            self.es_client = es_client
        else:
            self.es_client = EsClient()

        if query_builder:
            self.query_builder = query_builder
        else:
            self.query_builder = QueryBuilder()

    def _rrf(self, sparse_response: Response, dense_response: Response, k: int = 60) -> Response:
        id_to_product: dict[str, dict[str, Any]] = {}
        sparse_results: dict[str, float] = {}
        dense_results: dict[str, float] = {}

        for i, result in enumerate(sparse_response.results):
            product_id = result.product["product_id"]
            id_to_product[product_id] = result.product
            sparse_results[product_id] = 1 / (k + i + 1)
        for i, result in enumerate(dense_response.results):
            product_id = result.product["product_id"]
            id_to_product[product_id] = result.product
            dense_results[product_id] = 1 / (k + i + 1)

        results: list[Result] = []
        for product_id in sparse_results.keys() | dense_results.keys():
            score = sparse_results.get(product_id, 0) + dense_results.get(product_id, 0)
            result = Result(product=id_to_product[product_id], score=score)
            results.append(result)

        results = sorted(results, key=lambda result: (result.score, result.product["product_id"]), reverse=True)
        return Response(results=results, total_hits=len(results))

    def search(
        self,
        index_name: str,
        query: str,
        fields: list[str],
        is_synonym_expansion_enabled: bool = False,
        query_type: str = "combined_fields",
        product_ids: list[str] | None = None,
        sparse_boost: float = 1.0,
        dense_boost: float = 1.0,
        rrf: bool | int = False,
        size: int = 20,
    ):
        normalized_query = normalize_query(query)
        sparse_fields, dense_fields = split_fields(fields)
        sparse_query = None
        if sparse_fields:
            sparse_query = self.query_builder.build_sparse_search_query(
                query=normalized_query,
                fields=sparse_fields,
                query_type=query_type,
                boost=sparse_boost,
                is_synonym_expansion_enabled=is_synonym_expansion_enabled,
                product_ids=product_ids,
            )
        dense_query = None
        if normalized_query and dense_fields:
            dense_query = self.query_builder.build_dense_search_query(
                normalized_query,
                field=dense_fields[0],
                top_k=size,
                boost=dense_boost,
            )

        if not rrf:
            return self.es_client.search(
                index_name=index_name,
                query=sparse_query,
                knn_query=dense_query,
                size=size,
                explain=True,
            )

        sparse_response = self.es_client.search(
            index_name=index_name,
            query=sparse_query,
            knn_query=None,
            size=size,
            explain=True,
        )
        dense_response = self.es_client.search(
            index_name=index_name,
            query=None,
            knn_query=dense_query,
            size=size,
            explain=True,
        )
        if isinstance(rrf, bool):
            return self._rrf(sparse_response, dense_response)
        return self._rrf(sparse_response, dense_response, rrf)
