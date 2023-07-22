import json
from typing import Any

from torch import Tensor

from amazon_product_search.constants import DATA_DIR, HF
from amazon_product_search.core.cache import weak_lru_cache
from amazon_product_search.core.es.templates.template_loader import TemplateLoader
from amazon_product_search.core.synonyms.synonym_dict import SynonymDict
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder


class QueryBuilder:
    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self.synonym_dict = SynonymDict(data_dir)
        self.encoder: SBERTEncoder = SBERTEncoder(HF.JP_SLUKE_MEAN)
        self.template_loader = TemplateLoader()

    def match_all(self) -> dict[str, Any]:
        es_query_str = self.template_loader.load("match_all.j2").render()
        return json.loads(es_query_str)

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
            return self.match_all()

        synonyms = []
        if is_synonym_expansion_enabled:
            synonyms = self.synonym_dict.find_synonyms(query)

        query_match = json.loads(
            self.template_loader.load("query_match.j2").render(
                queries=[query, *synonyms],
                query_type=query_type,
                fields=fields,
                boost=boost,
            )
        )
        if not product_ids:
            return query_match
        return {
            "bool": {
                "should": [
                    query_match,
                ],
                "must": [
                    {
                        "terms": {
                            "product_id": product_ids,
                        },
                    },
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
        es_query_str = self.template_loader.load("dense.j2").render(
            query_vector=query_vector,
            field=field,
            k=top_k,
            num_candidates=100,
            boost=boost,
            product_ids=product_ids,
        )
        return json.loads(es_query_str)
