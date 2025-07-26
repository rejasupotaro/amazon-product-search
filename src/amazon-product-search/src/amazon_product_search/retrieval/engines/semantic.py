import json
import logging
import time

from amazon_product_search.es.templates.template_loader import TemplateLoader
from amazon_product_search.retrieval.core.types import FieldType, ProcessedQuery, RetrievalConfig, RetrievalResponse
from amazon_product_search.retrieval.engines.base import BaseRetrievalEngine

logger = logging.getLogger(__name__)


class SemanticRetrievalEngine(BaseRetrievalEngine):
    """Retrieval engine for dense vector similarity search."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, engine_name="semantic", **kwargs)
        self.template_loader = TemplateLoader()

    def retrieve(self, query: ProcessedQuery, config: RetrievalConfig) -> RetrievalResponse:
        """Perform semantic retrieval using dense vector similarity."""
        start_time = time.time()

        # Check if we have a query vector
        if not query.vector:
            logger.debug("No query vector found, skipping semantic retrieval")
            return RetrievalResponse(
                results=[],
                total_hits=0,
                engine_name=self.engine_name,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Get semantic fields
        semantic_fields = config.get_fields_by_type(FieldType.SEMANTIC)
        if not semantic_fields:
            logger.debug("No semantic fields found in config")
            return RetrievalResponse(
                results=[],
                total_hits=0,
                engine_name=self.engine_name,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Use first semantic field for now (could be enhanced to handle multiple)
        field = semantic_fields[0]

        # Build KNN query
        knn_query = self._build_semantic_query(
            query_vector=query.vector,
            field_name=field.name,
            top_k=config.window_size or config.size,
            filters=config.filters
        )

        # Execute search
        es_response = self.es_client.search(
            index_name=config.index_name,
            knn_query=knn_query,
            size=config.window_size or config.size,
            explain=config.enable_explain
        )

        processing_time = (time.time() - start_time) * 1000
        response = self._convert_es_response_to_retrieval_response(es_response, processing_time)
        response.metadata["vector_dim"] = len(query.vector)
        response.metadata["field_used"] = field.name

        logger.debug(f"Semantic retrieval found {len(response.results)} results in {processing_time:.2f}ms")
        return response

    def supports_fields(self, fields: list[str]) -> bool:
        """Check if we can handle semantic fields (vector fields)."""
        return any("vector" in field.lower() for field in fields)

    def _build_semantic_query(
        self,
        query_vector: list[float],
        field_name: str,
        top_k: int,
        filters: dict
    ) -> dict:
        """Build semantic search KNN query."""
        # Use existing template system
        es_query_str = self.template_loader.load("semantic.j2").render(
            query_vector=query_vector,
            field=field_name,
            k=top_k,
            num_candidates=top_k * 2,  # Use 2x candidates for better recall
            product_ids=filters.get("product_id")  # Support product_id filtering
        )
        return json.loads(es_query_str)
