from collections import defaultdict
from typing import Any

import pandas as pd

from amazon_product_search.constants import DATA_DIR
from amazon_product_search.nlp.tokenizer import Tokenizer


class QueryBuilder:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.synonyms_dict = self.load_synonym_dict()

    @staticmethod
    def load_synonym_dict(threshold: float = 0.6) -> dict[str, list[str]]:
        df = pd.read_csv(f"{DATA_DIR}/synonyms.csv")
        df = df[df["similarity"] > threshold]
        queries = df["query"].tolist()
        synonyms = df["title"].tolist()
        synonym_dict = defaultdict(list)
        for query, synonym in zip(queries, synonyms):
            synonym_dict[query].append(synonym)
        return synonym_dict

    def find_synonyms(self, query: str) -> list[str]:
        synonyms = []
        tokens = self.tokenizer.tokenize(query)
        for token in tokens:
            if token not in self.synonyms_dict:
                continue
            synonyms.extend(self.synonyms_dict[token])
        return synonyms

    def build_multimatch_search_query(
        self, query: str, fields: list[str], is_synonym_expansion_enabled: bool = False
    ) -> dict[str, Any]:
        """Build a multimatch ES query.

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

        synonyms = self.find_synonyms(query)
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

    def build_knn_search_query(self, query_vector: list[float], field: str, top_k: int) -> dict[str, Any]:
        """Build a KNN ES query from given conditions.

        Args:
            query_vector (list[float]): An encoded query vector.
            field (str): A field to examine.
            top_k (int): A number specifying how many results to return.

        Returns:
            dict[str, Any]: The constructed ES query.
        """
        return {
            "query_vector": query_vector,
            "field": field,
            "k": top_k,
            "num_candidates": 100,
        }


def build_multimatch_search_query(query: str, fields: list[str]) -> dict[str, Any]:
    """Build a multimatch ES query.

    Returns:
        dict[str, Any]: The constructed ES query.
    """
    if not query:
        return {
            "match_all": {},
        }

    return {
        "multi_match": {
            "query": query,
            "fields": fields,
            "operator": "or",
        }
    }


def build_knn_search_query(query_vector: list[float], field: str, top_k: int) -> dict[str, Any]:
    """Build a KNN ES query from given conditions.

    Args:
        query_vector (list[float]): An encoded query vector.
        field (str): A field to examine.
        top_k (int): A number specifying how many results to return.

    Returns:
        dict[str, Any]: The constructed ES query.
    """
    return {
        "query_vector": query_vector,
        "field": field,
        "k": top_k,
        "num_candidates": 100,
    }
