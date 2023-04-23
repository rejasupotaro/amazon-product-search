from typing import Any

from amazon_product_search.encoders.encoder import Encoder
from amazon_product_search.encoders.sbert_encoder import SBERTEncoder
from amazon_product_search.synonyms.synonym_dict import SynonymDict


class QueryBuilder:
    def __init__(self):
        self.synonym_dict = SynonymDict()
        self.encoder: Encoder = SBERTEncoder()

    def build_multimatch_search_query(
        self, query: str, fields: list[str], is_synonym_expansion_enabled: bool = False
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

        es_query = {
            "multi_match": {
                "query": query,
                "fields": fields,
                "operator": "or",
            }
        }
        if not is_synonym_expansion_enabled:
            return es_query

        synonyms = self.synonym_dict.find_synonyms(query)
        if not synonyms:
            return es_query

        match_clauses = []
        for q in [query] + synonyms:
            match_clauses.append(
                {
                    "multi_match": {
                        "query": q,
                        "fields": fields,
                        "operator": "or",
                    }
                }
            )
        return {
            "bool": {
                "should": match_clauses,
                "minimum_should_match": 1,
            }
        }

    def build_knn_search_query(self, query: str, field: str, top_k: int) -> dict[str, Any]:
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
