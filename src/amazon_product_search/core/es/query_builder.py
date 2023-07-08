from typing import Any

from torch import Tensor

from amazon_product_search.constants import DATA_DIR, HF
from amazon_product_search.core.cache import weak_lru_cache
from amazon_product_search.core.synonyms.synonym_dict import SynonymDict
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder


class QueryBuilder:
    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self.synonym_dict = SynonymDict(data_dir)
        self.encoder: SBERTEncoder = SBERTEncoder(HF.JP_SLUKE_MEAN)

    def _multi_match(self, query: str, fields: list[str], query_type: str, boost: float) -> dict[str, Any]:
        return {
            "multi_match": {
                "query": query,
                "type": query_type,
                "fields": fields,
                "operator": "and",
                "boost": boost,
            }
        }

    def _combined_fields(self, query: str, fields: list[str], boost: float) -> dict[str, Any]:
        return {
            "combined_fields": {
                "query": query,
                "fields": fields,
                "operator": "and",
                "boost": boost,
            }
        }

    def _simple_query_string(self, query: str, fields: list[str], boost: float) -> dict[str, Any]:
        return {
            "simple_query_string": {
                "query": query,
                "fields": fields,
                "default_operator": "and",
                "boost": boost,
            }
        }

    def _build_sparse_search_query(
        self, query_type: str, query: str, fields: list[str], boost: float
    ) -> dict[str, Any]:
        match query_type:
            case "cross_fields":
                return self._multi_match(query, fields, query_type="cross_fields", boost=boost)
            case "best_fields":
                return self._multi_match(query, fields, query_type="best_fields", boost=boost)
            case "combined_fields":
                return self._combined_fields(query, fields, boost)
            case "simple_query_string":
                return self._simple_query_string(query, fields, boost)
            case _:
                raise ValueError(f"Unknown query_type is given: {query_type}")

    def build_sparse_search_query(
        self,
        query: str,
        fields: list[str],
        query_type: str = "combined_fields",
        boost: float = 1.0,
        is_synonym_expansion_enabled: bool = False,
        product_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a multi-match ES query.

        Args:
            fields (list[str]): A list of fields to search.
            is_synonym_expansion_enabled: Expand the given query if True.

        Returns:
            dict[str, Any]: The constructed ES query.
        """
        if not query:
            return {
                "match_all": {},
            }

        synonyms = None
        if is_synonym_expansion_enabled:
            synonyms = self.synonym_dict.find_synonyms(query)

        terms_clause = None
        if product_ids:
            terms_clause = {
                "terms": {
                    "product_id": product_ids,
                },
            }

        if not synonyms:
            match_clause = self._build_sparse_search_query(query_type, query, fields, boost)
            if not terms_clause:
                return match_clause
            return {
                "bool": {
                    "should": [
                        match_clause,
                    ],
                    "must": [
                        terms_clause,
                    ],
                },
            }

        match_clauses = []
        for q in [query, *synonyms]:
            match_clauses.append(self._build_sparse_search_query(query_type, q, fields, boost))
        bool_clause = {
            "bool": {
                "should": match_clauses,
                "minimum_should_match": 1,
            }
        }
        if not terms_clause:
            return bool_clause
        return {
            "bool": {
                "should": [
                    bool_clause,
                ],
                "must": [
                    terms_clause,
                ],
            },
        }

    @weak_lru_cache(maxsize=128)
    def encode(self, query: str) -> Tensor:
        return self.encoder.encode(query)

    def build_dense_search_query(
        self,
        query: str,
        field: str,
        top_k: int,
        boost: float = 1.0,
        product_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a KNN ES query from given conditions.

        Args:
            query (str): A query to encode.
            field (str): A field to examine.
            top_k (int): A number specifying how many results to return.

        Returns:
            dict[str, Any]: The constructed ES query.
        """
        query_vector = self.encode(query).tolist()
        knn_clause = {
            "query_vector": query_vector,
            "field": field,
            "k": top_k,
            "num_candidates": 100,
            "boost": boost,
        }
        if not product_ids:
            return knn_clause
        knn_clause["filter"] = {
            "terms": {
                "product_id": product_ids,
            },
        }
        return knn_clause
