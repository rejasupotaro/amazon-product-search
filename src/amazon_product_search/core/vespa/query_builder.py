from typing import Any, cast

from amazon_product_search.core.cache import weak_lru_cache
from amazon_product_search.core.nlp.normalizer import normalize_query
from amazon_product_search.core.nlp.tokenizers import Tokenizer, locale_to_tokenizer
from amazon_product_search.core.retrieval.query_vector_cache import QueryVectorCache
from amazon_product_search.core.source import Locale
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder


class QueryBuilder:
    def __init__(
        self,
        locale: Locale,
        hf_model_name: str,
        vector_cache: QueryVectorCache | None = None,
    ) -> None:
        self.tokenizer: Tokenizer = locale_to_tokenizer(locale)
        self.encoder = SBERTEncoder(hf_model_name)
        if vector_cache is None:
            vector_cache = QueryVectorCache()
        self.vector_cache = vector_cache

    @weak_lru_cache(maxsize=128)
    def encode(self, query_str: str) -> list[float]:
        query_vector = self.vector_cache[query_str]
        if query_vector is not None:
            return query_vector
        return [float(v) for v in list(self.encoder.encode(query_str))]

    def build_query(
        self, query_str: str, rank_profile: str, size: int, is_semantic_search_enabled: bool
    ) -> dict[str, Any]:
        query_str = normalize_query(query_str)
        tokens = cast(list, self.tokenizer.tokenize(query_str))
        query_str = " ".join(tokens)

        query = {
            "query": query_str,
            "input.query(alpha)": 0.5,
            "ranking.profile": rank_profile,
            "hits": size,
        }
        if is_semantic_search_enabled:
            query[
                "yql"
            ] = """
            select
                *
            from
                product
            where
                userQuery()
                or ({targetHits:100, approximate:true}nearestNeighbor(product_vector, query_vector))
            """
            query["input.query(query_vector)"] = self.encode(query_str)
        else:
            query[
                "yql"
            ] = """
            select
                *
            from
                product
            where
                userQuery()
            """
        return query

    def build_lexical_search_query(self, query_str: str, size: int) -> dict[str, Any]:
        return self.build_query(query_str, rank_profile="lexical", size=size, is_semantic_search_enabled=False)

    def build_semantic_search_query(self, query_str: str, size: int) -> dict[str, Any]:
        return self.build_query(query_str, rank_profile="semantic", size=size, is_semantic_search_enabled=True)

    def build_hybrid_search_query(self, query_str: str, size: int) -> dict[str, Any]:
        return self.build_query(query_str, rank_profile="hybrid", size=size, is_semantic_search_enabled=True)
