import logging
from typing import TYPE_CHECKING, Any

from data_source import Locale

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.retrieval.core.types import FusionConfig
from amazon_product_search.retrieval.engines.lexical import LexicalRetrievalEngine
from amazon_product_search.retrieval.engines.semantic import SemanticRetrievalEngine
from amazon_product_search.retrieval.fusion.base import FlexibleResultFuser
from amazon_product_search.retrieval.pipeline import RetrievalPipeline
from amazon_product_search.retrieval.processors.base import (
    BaseQueryProcessor,
    ProcessorChain,
    SynonymExpandingProcessor,
)
from amazon_product_search.retrieval.processors.semantic import SemanticQueryProcessor
from amazon_product_search.retrieval.rank_fusion import RankFusion, fuse
from amazon_product_search.retrieval.resources.manager import SharedResourceManager
from amazon_product_search.retrieval.response import Response
from amazon_product_search.synonyms.synonym_dict import SynonymDict

if TYPE_CHECKING:
    from amazon_product_search.retrieval.core.protocols import QueryProcessor

logger = logging.getLogger(__name__)


def split_fields(fields: list[str]) -> tuple[list[str], list[str]]:
    """Convert a given list of fields into a tuple of (lexical_fields, semantic_fields)

    DEPRECATED: This function is kept for backward compatibility.
    Use FieldType enum and SearchField objects in the new architecture.

    Field names containing "vector" will be considered semantic_fields.

    Args:
        fields (list[str]): A list of fields.

    Returns:
        tuple[list[str], list[str]]: A tuple of (lexical_fields, semantic_fields)
    """
    logger.warning("split_fields() is deprecated. Use FieldType enum and SearchField objects instead.")

    lexical_fields: list[str] = []
    semantic_fields: list[str] = []
    for field in fields:
        (semantic_fields if "vector" in field else lexical_fields).append(field)
    return lexical_fields, semantic_fields


class Retriever:
    """
    Modernized retrieval system with backward compatibility.

    This class now uses the new modular architecture internally while maintaining
    the same API for backward compatibility.
    """

    def __init__(
        self,
        locale: Locale,
        es_client: EsClient | None = None,
        query_builder: Any | None = None,  # Kept for compatibility but not used
        synonym_dict: SynonymDict | None = None,
        resource_manager: SharedResourceManager | None = None,
        use_new_architecture: bool = True
    ) -> None:
        """Initialize retriever with new modular architecture.

        Args:
            locale: Locale for query processing
            es_client: Elasticsearch client (optional)
            query_builder: Legacy parameter, ignored in new architecture
            synonym_dict: Optional synonym dictionary
            resource_manager: Optional shared resource manager
            use_new_architecture: Whether to use new modular architecture (default: True)
        """
        self.locale = locale
        self.use_new_architecture = use_new_architecture

        if es_client:
            self.es_client = es_client
        else:
            self.es_client = EsClient()

        if not use_new_architecture:
            # Fallback to legacy implementation
            from amazon_product_search.es.query_builder import QueryBuilder
            self.query_builder = query_builder or QueryBuilder(locale)
            logger.warning("Using legacy retrieval architecture. Consider upgrading to new architecture.")
            return

        # Initialize new architecture components
        if resource_manager is None:
            resource_manager = SharedResourceManager()
        self.resource_manager = resource_manager

        # Set up query processing pipeline
        base_processor = BaseQueryProcessor(locale)
        processors: list[QueryProcessor] = [base_processor]

        if synonym_dict:
            synonym_processor = SynonymExpandingProcessor(base_processor, synonym_dict)
            processors.append(synonym_processor)

        # Add semantic processing
        semantic_processor = SemanticQueryProcessor(
            base_processor=processors[-1],
            resource_manager=resource_manager
        )
        processors.append(semantic_processor)

        self.query_processor = ProcessorChain(processors)

        # Set up retrieval engines
        self.lexical_engine = LexicalRetrievalEngine(es_client=self.es_client)
        self.semantic_engine = SemanticRetrievalEngine(es_client=self.es_client)
        self.retrieval_engines = [self.lexical_engine, self.semantic_engine]

        # Set up result fusion
        fusion_config = FusionConfig(method="weighted_sum", normalization="min_max")
        self.result_fuser = FlexibleResultFuser(fusion_config)

        # Create pipeline
        self.pipeline = RetrievalPipeline(
            query_processor=self.query_processor,
            retrieval_engines=self.retrieval_engines,
            result_fuser=self.result_fuser,
            resource_manager=resource_manager
        )

        logger.info(f"Initialized Retriever with new modular architecture for locale: {locale}")

    def search(
        self,
        index_name: str,
        query: str,
        fields: list[str],
        enable_synonym_expansion: bool = False,
        product_ids: list[str] | None = None,
        lexical_boost: float = 1.0,
        semantic_boost: float = 1.0,
        size: int = 20,
        window_size: int | None = None,
        rank_fusion: RankFusion | None = None,
    ) -> Response:
        """Search with both new and legacy architecture support.

        Args:
            index_name: Elasticsearch index name
            query: Search query string
            fields: List of field names to search
            enable_synonym_expansion: Whether to expand synonyms
            product_ids: Optional list of product IDs to filter
            lexical_boost: Boost factor for lexical results
            semantic_boost: Boost factor for semantic results
            size: Number of results to return
            window_size: Window size for initial retrieval
            rank_fusion: Legacy rank fusion config (ignored in new architecture)

        Returns:
            Response: Search results in legacy format
        """
        if not self.use_new_architecture:
            # Use legacy implementation
            return self._search_legacy(
                index_name, query, fields, enable_synonym_expansion,
                product_ids, lexical_boost, semantic_boost, size, window_size, rank_fusion
            )

        # Use new architecture via pipeline
        return self.pipeline.search_legacy(
            index_name=index_name,
            query=query,
            fields=fields,
            enable_synonym_expansion=enable_synonym_expansion,
            product_ids=product_ids,
            lexical_boost=lexical_boost,
            semantic_boost=semantic_boost,
            size=size,
            window_size=window_size,
            rank_fusion=rank_fusion
        )

    def _search_legacy(
        self,
        index_name: str,
        query: str,
        fields: list[str],
        enable_synonym_expansion: bool = False,
        product_ids: list[str] | None = None,
        lexical_boost: float = 1.0,
        semantic_boost: float = 1.0,
        size: int = 20,
        window_size: int | None = None,
        rank_fusion: RankFusion | None = None,
    ) -> Response:
        """Legacy search implementation for backward compatibility."""
        from amazon_product_search.nlp.normalizer import normalize_query

        normalized_query = normalize_query(query)
        lexical_fields, semantic_fields = split_fields(fields)
        if window_size is None:
            window_size = size

        if not rank_fusion:
            rank_fusion = RankFusion()

        lexical_query = None
        if lexical_fields:
            lexical_query = self.query_builder.build_lexical_search_query(
                query=normalized_query,
                fields=lexical_fields,
                enable_synonym_expansion=enable_synonym_expansion,
                product_ids=product_ids,
            )
        semantic_query = None
        if normalized_query and semantic_fields:
            semantic_query = self.query_builder.build_semantic_search_query(
                normalized_query,
                field=semantic_fields[0],
                top_k=window_size,
                product_ids=product_ids,
            )

        if lexical_query:
            lexical_response = self.es_client.search(
                index_name=index_name,
                query=lexical_query,
                knn_query=None,
                size=window_size,
                explain=True,
            )
        else:
            lexical_response = Response(results=[], total_hits=0)
        if semantic_query:
            semantic_response = self.es_client.search(
                index_name=index_name,
                query=None,
                knn_query=semantic_query,
                size=window_size,
                explain=True,
            )
        else:
            semantic_response = Response(results=[], total_hits=0)

        return fuse(query, lexical_response, semantic_response, lexical_boost, semantic_boost, rank_fusion, size)

    def add_retrieval_engine(self, engine: Any) -> None:
        """Add a new retrieval engine to the pipeline."""
        if not self.use_new_architecture:
            raise NotImplementedError("Adding engines is only supported in the new architecture")

        self.retrieval_engines.append(engine)
        self.pipeline = RetrievalPipeline(
            query_processor=self.query_processor,
            retrieval_engines=self.retrieval_engines,
            result_fuser=self.result_fuser,
            resource_manager=self.resource_manager
        )
        logger.info(f"Added retrieval engine: {engine.__class__.__name__}")

    def add_post_processor(self, processor: Any) -> None:
        """Add a post-processor to the pipeline."""
        if not self.use_new_architecture:
            raise NotImplementedError("Adding post-processors is only supported in the new architecture")

        self.pipeline.add_post_processor(processor)

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the current pipeline configuration."""
        if not self.use_new_architecture:
            return {"architecture": "legacy", "locale": self.locale}

        return {
            "architecture": "modular",
            "locale": self.locale,
            **self.pipeline.get_pipeline_info()
        }
