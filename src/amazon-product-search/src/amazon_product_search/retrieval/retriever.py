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
from amazon_product_search.retrieval.rank_fusion import RankFusion
from amazon_product_search.retrieval.resources.manager import SharedResourceManager
from amazon_product_search.retrieval.response import Response
from amazon_product_search.synonyms.synonym_dict import SynonymDict

if TYPE_CHECKING:
    from amazon_product_search.retrieval.core.protocols import QueryProcessor

logger = logging.getLogger(__name__)



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
        synonym_dict: SynonymDict | None = None,
        resource_manager: SharedResourceManager | None = None,
    ) -> None:
        """Initialize retriever with modular architecture.

        Args:
            locale: Locale for query processing
            es_client: Elasticsearch client (optional)
            synonym_dict: Optional synonym dictionary
            resource_manager: Optional shared resource manager
        """
        self.locale = locale

        if es_client:
            self.es_client = es_client
        else:
            self.es_client = EsClient()

        # Initialize architecture components
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
        """Search using the modular architecture.

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
            rank_fusion: Legacy rank fusion config (ignored)

        Returns:
            Response: Search results in legacy format
        """
        # Use modular architecture via pipeline
        # Convert legacy parameters to new format
        from amazon_product_search.retrieval.core.types import FieldType, RetrievalConfig, SearchField

        search_fields = []
        for field_name in fields:
            is_semantic = "vector" in field_name.lower() or "embedding" in field_name.lower()
            field_type = FieldType.SEMANTIC if is_semantic else FieldType.LEXICAL
            search_fields.append(SearchField(name=field_name, field_type=field_type))

        config = RetrievalConfig(
            index_name=index_name,
            fields=search_fields,
            size=size,
            window_size=window_size,
            enable_synonyms=enable_synonym_expansion,
            filters={"product_id": product_ids} if product_ids else {}
        )

        # Set up fusion weights
        fusion_weights = {
            "lexical": lexical_boost,
            "semantic": semantic_boost
        }

        # Execute new pipeline
        response = self.pipeline.search(query, config, fusion_weights)

        # Convert back to legacy format
        from amazon_product_search.retrieval.response import Response
        return Response(results=response.results, total_hits=response.total_hits)


    def add_retrieval_engine(self, engine: Any) -> None:
        """Add a new retrieval engine to the pipeline."""
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
        self.pipeline.add_post_processor(processor)

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the current pipeline configuration."""
        return {
            "architecture": "modular",
            "locale": self.locale,
            **self.pipeline.get_pipeline_info()
        }
