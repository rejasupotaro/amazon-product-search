from collections import defaultdict
from typing import Any

import pandas as pd

from amazon_product_search.constants import DATA_DIR
from amazon_product_search.nlp.encoder import Encoder
from amazon_product_search.nlp.tokenizer import Tokenizer


class QueryBuilder:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.synonyms_dict = self.load_synonym_dict()
        self.encoder = Encoder()

    @staticmethod
    def load_synonym_dict(threshold: float = 0.6) -> dict[str, list[str]]:
        """Load the synonym file and convert it into a dict for lookup.

        Args:
            threshold (float, optional): A threshold for the confidence (similarity) of synonyms. Defaults to 0.6.

        Returns:
            dict[str, list[str]]: The converted synonym dict.
        """
        df = pd.read_csv(f"{DATA_DIR}/synonyms.csv")
        df = df[df["similarity"] > threshold]
        queries = df["query"].tolist()
        synonyms = df["title"].tolist()
        synonym_dict = defaultdict(list)
        for query, synonym in zip(queries, synonyms):
            synonym_dict[query].append(synonym)
        return synonym_dict

    def find_synonyms(self, query: str) -> list[str]:
        """Return a list of synonyms for a given query.

        Args:
            query (str): A query to expand.

        Returns:
            list[str]: A list of synonyms.
        """
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

    def build_knn_search_query(self, query: str, field: str, top_k: int) -> dict[str, Any]:
        """Build a KNN ES query from given conditions.

        Args:
            query (str): A query to encode.
            field (str): A field to examine.
            top_k (int): A number specifying how many results to return.

        Returns:
            dict[str, Any]: The constructed ES query.
        """
        query_vector = self.encoder.encode(query, show_progress_bar=False)
        return {
            "query_vector": query_vector,
            "field": field,
            "k": top_k,
            "num_candidates": 100,
        }
