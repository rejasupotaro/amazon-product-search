from typing import Any, Literal, cast

from amazon_product_search.core.cache import weak_lru_cache
from amazon_product_search.core.nlp.normalizer import normalize_query
from amazon_product_search.core.nlp.tokenizers import Tokenizer, locale_to_tokenizer
from amazon_product_search.core.retrieval.query_vector_cache import QueryVectorCache
from amazon_product_search.core.source import Locale
from amazon_product_search.core.synonyms.synonym_dict import SynonymDict
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder

Operator = Literal["and", "weakAnd"]


class QueryBuilder:
    def __init__(
        self,
        locale: Locale,
        hf_model_name: str,
        synonym_dict: SynonymDict | None = None,
        vector_cache: QueryVectorCache | None = None,
    ) -> None:
        self.tokenizer: Tokenizer = locale_to_tokenizer(locale)
        self.encoder = SBERTEncoder(hf_model_name)
        if synonym_dict is None:
            synonym_dict = SynonymDict(locale)
        self.synonym_dict = synonym_dict
        if vector_cache is None:
            vector_cache = QueryVectorCache()
        self.vector_cache = vector_cache

    @weak_lru_cache(maxsize=128)
    def encode(self, query_str: str) -> list[float]:
        query_vector = self.vector_cache[query_str]
        if query_vector is not None:
            return query_vector
        return self.encoder.encode(query_str).tolist()

    def _build_text_matching_query(self, tokens: list[str], fields: list[str], operator: Operator) -> str:
        if operator == "weakAnd":
            conditions = []
            for token, synonyms in self.synonym_dict.expand_synonyms(" ".join(tokens)):
                for field in fields:
                    conditions.append(f"{field} contains '{token}'")
                    for synonym in synonyms:
                        conditions.append(f"{field} contains '{synonym}'")
            return f"weakAnd({', '.join(conditions)})"

        and_conditions = []
        for token, synonyms in self.synonym_dict.expand_synonyms(" ".join(tokens)):
            or_conditions = []
            for field in fields:
                if synonyms:
                    equiv_query = ", ".join([f"'{token}'" for token in [token, *synonyms]])
                    or_conditions.append(f"{field} contains equiv({equiv_query})")
                else:
                    or_conditions.append(f"{field} contains '{token}'")
            and_conditions.append(f"({' OR '.join(or_conditions)})")
        return " AND ".join(and_conditions)

    def build_query(
        self,
        query_str: str,
        rank_profile: str,
        size: int,
        is_semantic_search_enabled: bool = False,
        fields: list[str] | None = None,
        operator: Operator = "and",
        alpha: float = 0.5,
        title_weight: float = 1.0,
        brand_weight: float = 1.0,
        color_weight: float = 1.0,
        bullet_point_weight: float = 1.0,
        description_weight: float = 1.0,
    ) -> dict[str, Any]:
        query_str = normalize_query(query_str)
        tokens = cast(list, self.tokenizer.tokenize(query_str))
        query_str = " ".join(tokens)

        if not fields:
            fields = ["default"]
        text_matching_query = self._build_text_matching_query(tokens, fields, operator)

        query = {
            "query": query_str,
            "input.query(alpha)": alpha,
            "input.query(title_weight)": title_weight,
            "input.query(brand_weight)": brand_weight,
            "input.query(color_weight)": color_weight,
            "input.query(bullet_point_weight)": bullet_point_weight,
            "input.query(description_weight)": description_weight,
            "ranking.profile": rank_profile,
            "hits": size,
        }
        if is_semantic_search_enabled:
            query[
                "yql"
            ] = f"""
            select
                *
            from
                product
            where
                {text_matching_query}
                or ({{targetHits:{size}, approximate:true}}nearestNeighbor(product_vector, query_vector))
            """
            query["input.query(query_vector)"] = self.encode(query_str)
        else:
            query[
                "yql"
            ] = f"""
            select
                *
            from
                product
            where
                {text_matching_query}
            """
        return query

    def build_lexical_search_query(
        self,
        query_str: str,
        size: int,
        fields: list[str] | None = None,
        operator: Operator = "and",
        title_weight: float = 1.0,
        brand_weight: float = 1.0,
        color_weight: float = 1.0,
        bullet_point_weight: float = 1.0,
        description_weight: float = 1.0,
    ) -> dict[str, Any]:
        return self.build_query(
            query_str,
            rank_profile="lexical",
            size=size,
            is_semantic_search_enabled=False,
            fields=fields,
            operator=operator,
            title_weight=title_weight,
            brand_weight=brand_weight,
            color_weight=color_weight,
            bullet_point_weight=bullet_point_weight,
            description_weight=description_weight,
        )

    def build_semantic_search_query(
        self,
        query_str: str,
        size: int,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.build_query(
            query_str,
            rank_profile="semantic",
            size=size,
            is_semantic_search_enabled=True,
            fields=fields,
        )

    def build_hybrid_search_query(
        self, query_str: str, size: int, fields: list[str] | None = None, operator: Operator = "and", alpha: float = 0.5
    ) -> dict[str, Any]:
        return self.build_query(
            query_str,
            rank_profile="hybrid",
            size=size,
            is_semantic_search_enabled=True,
            fields=fields,
            operator=operator,
            alpha=alpha,
        )
