from amazon_product_search.core.es.es_client import EsClient
from amazon_product_search.core.es.query_builder import QueryBuilder
from amazon_product_search.core.es.response import Response
from amazon_product_search.core.nlp.normalizer import normalize_query
from amazon_product_search.core.retrieval.rank_fusion import RankFusion, fuse
from amazon_product_search.core.source import Locale


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
    def __init__(
        self, locale: Locale, es_client: EsClient | None = None, query_builder: QueryBuilder | None = None
    ) -> None:
        if es_client:
            self.es_client = es_client
        else:
            self.es_client = EsClient()

        if query_builder:
            self.query_builder = query_builder
        else:
            self.query_builder = QueryBuilder(locale)

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
        size: int = 20,
        window_size: int | None = None,
        rank_fusion: RankFusion | None = None,
    ) -> Response:
        normalized_query = normalize_query(query)
        sparse_fields, dense_fields = split_fields(fields)
        if window_size is None:
            window_size = size

        if not rank_fusion:
            rank_fusion = RankFusion()

        sparse_query = None
        if sparse_fields:
            sparse_query = self.query_builder.build_sparse_search_query(
                query=normalized_query,
                fields=sparse_fields,
                query_type=query_type,
                # Boost should be 1.0 if fuser == "own" because boost will be applied later.
                boost=1.0 if rank_fusion.fuser == "own" else sparse_boost,
                is_synonym_expansion_enabled=is_synonym_expansion_enabled,
                product_ids=product_ids,
            )
        dense_query = None
        if normalized_query and dense_fields:
            dense_query = self.query_builder.build_dense_search_query(
                normalized_query,
                field=dense_fields[0],
                top_k=window_size,
                # Boost should be 1.0 if fuser == "own" because boost will be applied later.
                boost=1.0 if rank_fusion.fuser == "own" else dense_boost,
                product_ids=product_ids,
            )

        if rank_fusion.fuser == "search_engine":
            rank = None
            if rank_fusion.normalization_strategy == "rrf":
                rank = {
                    "rrf": {
                        "window_size": window_size,
                    }
                }
            return self.es_client.search(
                index_name=index_name,
                query=sparse_query,
                knn_query=dense_query,
                rank=rank,
                size=size,
                explain=True,
            )

        if sparse_query:
            sparse_response = self.es_client.search(
                index_name=index_name,
                query=sparse_query,
                knn_query=None,
                size=window_size,
                explain=True,
            )
        else:
            sparse_response = Response(results=[], total_hits=0)
        if dense_query:
            dense_response = self.es_client.search(
                index_name=index_name,
                query=None,
                knn_query=dense_query,
                size=window_size,
                explain=True,
            )
        else:
            dense_response = Response(results=[], total_hits=0)

        return fuse(query, sparse_response, dense_response, sparse_boost, dense_boost, rank_fusion, size)
