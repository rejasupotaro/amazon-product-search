import logging
import time

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.retrieval.core.protocols import RetrievalEngine
from amazon_product_search.retrieval.core.types import (
    ProcessedQuery,
    RetrievalConfig,
    RetrievalResponse,
)
from amazon_product_search.retrieval.response import Response

logger = logging.getLogger(__name__)


class BaseRetrievalEngine(RetrievalEngine):
    """Base implementation for retrieval engines."""

    def __init__(self, es_client: EsClient | None = None, engine_name: str = "base"):
        if es_client:
            self.es_client = es_client
        else:
            self.es_client = EsClient()
        self.engine_name = engine_name

    def retrieve(self, query: ProcessedQuery, config: RetrievalConfig) -> RetrievalResponse:
        """Base retrieve implementation - should be overridden by subclasses."""
        start_time = time.time()

        # Default to empty response
        response = RetrievalResponse(
            results=[],
            total_hits=0,
            engine_name=self.engine_name,
            processing_time_ms=(time.time() - start_time) * 1000
        )

        return response

    def supports_fields(self, fields: list[str]) -> bool:
        """Base implementation - should be overridden."""
        return False

    def _convert_es_response_to_retrieval_response(
        self,
        es_response: Response,
        processing_time_ms: float = 0.0
    ) -> RetrievalResponse:
        """Convert ES Response to RetrievalResponse."""
        return RetrievalResponse(
            results=es_response.results,
            total_hits=es_response.total_hits,
            engine_name=self.engine_name,
            processing_time_ms=processing_time_ms
        )
