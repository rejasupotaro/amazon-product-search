from amazon_product_search.core.es.es_client import EsClient
from amazon_product_search.core.es.query_builder import QueryBuilder
from amazon_product_search.core.nlp.normalizer import normalize_query
from amazon_product_search.core.retrieval.rank_fusion import RankFusion, fuse
from amazon_product_search.core.retrieval.response import Response
from amazon_product_search.core.source import Locale


def split_fields(fields: list[str]) -> tuple[list[str], list[str]]:
    """Convert a given list of fields into a tuple of (lexical_fields, semantic_fields)

    Field names containing "vector" will be considered semantic_fields.

    Args:
        fields (list[str]): A list of fields.

    Returns:
        tuple[list[str], list[str]]: A tuple of (lexical_fields, semantic_fields)
    """
    lexical_fields: list[str] = []
    semantic_fields: list[str] = []
    for field in fields:
        (semantic_fields if "vector" in field else lexical_fields).append(field)
    return lexical_fields, semantic_fields


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
        enable_synonym_expansion: bool = False,
        product_ids: list[str] | None = None,
        lexical_boost: float = 1.0,
        semantic_boost: float = 1.0,
        size: int = 20,
        window_size: int | None = None,
        rank_fusion: RankFusion | None = None,
    ) -> Response:
        normalized_query = normalize_query(query)
        lexical_fields, semantic_fields = split_fields(fields)
        if window_size is None:
            window_size = size

        if not rank_fusion:
            rank_fusion = RankFusion()

        lexical_query = None
        if lexical_fields:
            lexical_query = self.query_builder.build_lexical_search_query(
                query=normalized_query,
                fields=lexical_fields,
                enable_synonym_expansion=enable_synonym_expansion,
                product_ids=product_ids,
            )
        semantic_query = None
        if normalized_query and semantic_fields:
            semantic_query = self.query_builder.build_semantic_search_query(
                normalized_query,
                field=semantic_fields[0],
                top_k=window_size,
                product_ids=product_ids,
            )

        if lexical_query:
            lexical_response = self.es_client.search(
                index_name=index_name,
                query=lexical_query,
                knn_query=None,
                size=window_size,
                explain=True,
            )
        else:
            lexical_response = Response(results=[], total_hits=0)
        if semantic_query:
            semantic_response = self.es_client.search(
                index_name=index_name,
                query=None,
                knn_query=semantic_query,
                size=window_size,
                explain=True,
            )
        else:
            semantic_response = Response(results=[], total_hits=0)

        return fuse(query, lexical_response, semantic_response, lexical_boost, semantic_boost, rank_fusion, size)
