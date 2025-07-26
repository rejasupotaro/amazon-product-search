import json
import logging
import time

from amazon_product_search.es.templates.template_loader import TemplateLoader
from amazon_product_search.retrieval.core.types import FieldType, ProcessedQuery, RetrievalConfig, RetrievalResponse
from amazon_product_search.retrieval.engines.base import BaseRetrievalEngine

logger = logging.getLogger(__name__)


class LexicalRetrievalEngine(BaseRetrievalEngine):
    """Retrieval engine for lexical/keyword-based search."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, engine_name="lexical", **kwargs)
        self.template_loader = TemplateLoader()

    def retrieve(self, query: ProcessedQuery, config: RetrievalConfig) -> RetrievalResponse:
        """Perform lexical retrieval using BM25/keyword matching."""
        start_time = time.time()

        # Get lexical fields
        lexical_fields = config.get_fields_by_type(FieldType.LEXICAL)
        if not lexical_fields:
            logger.debug("No lexical fields found in config")
            return RetrievalResponse(
                results=[],
                total_hits=0,
                engine_name=self.engine_name,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Build queries (including synonym expansions)
        queries = [query.normalized]
        if query.synonyms:
            queries.extend(query.synonyms)

        # Prepare field list with weights
        weighted_fields = [
            f"{field.name}^{field.weight}" for field in lexical_fields
        ]

        # Build ES query
        es_query = self._build_lexical_query(
            queries=queries,
            fields=weighted_fields,
            filters=config.filters
        )

        # Execute search
        es_response = self.es_client.search(
            index_name=config.index_name,
            query=es_query,
            size=config.window_size or config.size,
            explain=config.enable_explain
        )

        processing_time = (time.time() - start_time) * 1000
        response = self._convert_es_response_to_retrieval_response(es_response, processing_time)
        response.metadata["query_variations"] = len(queries)

        logger.debug(f"Lexical retrieval found {len(response.results)} results in {processing_time:.2f}ms")
        return response

    def supports_fields(self, fields: list[str]) -> bool:
        """Check if we can handle lexical fields (non-vector fields)."""
        return all("vector" not in field.lower() and "embedding" not in field.lower() for field in fields)

    def _build_lexical_query(self, queries: list[str], fields: list[str], filters: dict) -> dict:
        """Build lexical search query using templates."""
        # Use the existing template system
        query_match = json.loads(
            self.template_loader.load("lexical.j2").render(
                queries=queries[:10],  # Limit to 10 queries
                fields=fields,
                operator="and",
                enable_phrase_match_boost=False,
            )
        )

        # Add filters if specified
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {key: value}})
                else:
                    filter_clauses.append({"term": {key: value}})

            if filter_clauses:
                return {
                    "bool": {
                        "must": [query_match],
                        "filter": filter_clauses
                    }
                }

        return query_match
