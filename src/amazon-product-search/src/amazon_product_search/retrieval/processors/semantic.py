import logging

from amazon_product_search.cache import weak_lru_cache
from amazon_product_search.constants import HF
from amazon_product_search.retrieval.core.protocols import QueryProcessor, ResourceManager
from amazon_product_search.retrieval.core.types import ProcessedQuery, RetrievalConfig
from amazon_product_search.retrieval.query_vector_cache import QueryVectorCache
from dense_retrieval.encoders import SBERTEncoder

logger = logging.getLogger(__name__)


class SemanticQueryProcessor(QueryProcessor):
    """Query processor that adds dense vector representations."""

    def __init__(
        self,
        base_processor: QueryProcessor,
        resource_manager: ResourceManager | None = None,
        model_name: str = HF.JP_SLUKE_MEAN,
        vector_cache: QueryVectorCache | None = None
    ):
        self.base_processor = base_processor
        self.resource_manager = resource_manager
        self.model_name = model_name

        # Initialize encoder directly if no resource manager
        if not resource_manager:
            self.encoder = SBERTEncoder(model_name)
        else:
            self.encoder = None

        if vector_cache is None:
            vector_cache = QueryVectorCache()
        self.vector_cache = vector_cache

    def process(self, raw_query: str, config: RetrievalConfig) -> ProcessedQuery:
        """Process query and add dense vector representation."""
        query = self.base_processor.process(raw_query, config)

        if not query.normalized:
            return query

        # Get vector representation
        query.vector = self._encode_query(query.normalized)
        query.metadata["vector_dim"] = len(query.vector) if query.vector else 0

        return query

    @weak_lru_cache(maxsize=128)
    def _encode_query(self, text: str) -> list[float]:
        """Encode query text to vector."""
        # Check cache first
        cached_vector = self.vector_cache[text]
        if cached_vector is not None:
            logger.debug(f"Using cached vector for query: {text}")
            return cached_vector

        # Get encoder (from resource manager or direct)
        encoder = self.resource_manager.get_encoder(self.model_name) if self.resource_manager else self.encoder

        # Encode and cache
        vector = encoder.encode(text).tolist()
        logger.debug(f"Encoded query '{text}' to {len(vector)}-dim vector")

        return vector


class SparseQueryProcessor(QueryProcessor):
    """Query processor that adds sparse vector representations (e.g., for SPLADE)."""

    def __init__(
        self,
        base_processor: QueryProcessor,
        resource_manager: ResourceManager | None = None,
        model_name: str = HF.JP_SPLADE
    ):
        self.base_processor = base_processor
        self.resource_manager = resource_manager
        self.model_name = model_name

        # Note: SPLADE implementation would need to be adapted here
        # For now, this is a placeholder for future sparse vector support

    def process(self, raw_query: str, config: RetrievalConfig) -> ProcessedQuery:
        """Process query and add sparse vector representation."""
        query = self.base_processor.process(raw_query, config)

        # TODO: Implement SPLADE encoding
        # query.sparse_vector = self._encode_sparse(query.normalized)

        return query
