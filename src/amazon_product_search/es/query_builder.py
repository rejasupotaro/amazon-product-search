from typing import Any

from amazon_product_search.constants import DATA_DIR
from amazon_product_search.constants import HF
from amazon_product_search.synonyms.synonym_dict import SynonymDict
from amazon_product_search_dense_retrieval.encoders import Encoder, SBERTEncoder


class QueryBuilder:
    def __init__(self, data_dir: str = DATA_DIR):
        self.synonym_dict = SynonymDict(data_dir)
        self.encoder: Encoder = SBERTEncoder(HF.JP_SBERT)

    def _multi_match(self, query: str, fields: list[str], match_type: str = "cross_fields") -> dict[str, Any]:
        return {
            "multi_match": {
                "query": query,
                "type": match_type,
                "fields": fields,
                "operator": "and",
            }
        }

    def _combined_fields(self, query: str, fields: list[str]) -> dict[str, Any]:
        return {
            "combined_fields": {
                "query": query,
                "fields": fields,
                "operator": "and",
            }
        }

    def _simple_query_string(self, query: str, fields: list[str]) -> dict[str, Any]:
        return {
            "simple_query_string": {
                "query": query,
                "fields": fields,
                "default_operator": "and",
            }
        }

    def _build_sparse_search_query(self, query_type: str, query: str, fields: list[str]) -> dict[str, Any]:
        match query_type:
            case "multi_match":
                return self._multi_match(query, fields)
            case "combined_fields":
                return self._combined_fields(query, fields)
            case "simple_query_string":
                return self._simple_query_string(query, fields)
            case _:
                raise ValueError(f"Unknown query_type is given: {query_type}")

    def build_sparse_search_query(
        self, query: str,
        fields: list[str],
        query_type: str = "multi_match",
        is_synonym_expansion_enabled: bool = False,
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

        es_query = self._build_sparse_search_query(query_type, query, fields)
        if not is_synonym_expansion_enabled:
            return es_query

        synonyms = self.synonym_dict.find_synonyms(query)
        if not synonyms:
            return es_query

        match_clauses = []
        for q in [query, *synonyms]:
            match_clauses.append(self._build_sparse_search_query(query_type, q, fields))
        return {
            "bool": {
                "should": match_clauses,
                "minimum_should_match": 1,
            }
        }

    def build_dense_search_query(self, query: str, field: str, top_k: int) -> dict[str, Any]:
        """Build a KNN ES query from given conditions.

        Args:
            query (str): A query to encode.
            field (str): A field to examine.
            top_k (int): A number specifying how many results to return.

        Returns:
            dict[str, Any]: The constructed ES query.
        """
        query_vector = self.encoder.encode(query)
        return {
            "query_vector": query_vector,
            "field": field,
            "k": top_k,
            "num_candidates": 100,
        }
