"""Factory functions for creating configured retrieval components and pipelines."""

import logging
from typing import TYPE_CHECKING, Any

from data_source import Locale

from amazon_product_search.constants import HF
from amazon_product_search.es.es_client import EsClient
from amazon_product_search.retrieval.core.types import FieldType, FusionConfig, SearchField
from amazon_product_search.retrieval.engines.lexical import LexicalRetrievalEngine
from amazon_product_search.retrieval.engines.semantic import SemanticRetrievalEngine
from amazon_product_search.retrieval.fusion.base import FlexibleResultFuser
from amazon_product_search.retrieval.pipeline import RetrievalPipeline
from amazon_product_search.retrieval.processors.base import (
    BaseQueryProcessor,
    ProcessorChain,
    SynonymExpandingProcessor,
)
from amazon_product_search.retrieval.processors.reranking import ResourceAwareRerankerProcessor
from amazon_product_search.retrieval.processors.semantic import SemanticQueryProcessor
from amazon_product_search.retrieval.resources.manager import SharedResourceManager
from amazon_product_search.retrieval.retriever import Retriever
from amazon_product_search.synonyms.synonym_dict import SynonymDict

if TYPE_CHECKING:
    from amazon_product_search.retrieval.core.protocols import QueryProcessor, RetrievalEngine

logger = logging.getLogger(__name__)


def create_search_fields(field_names: list[str], weights: dict[str, float] | None = None) -> list[SearchField]:
    """Create SearchField objects from field names with automatic type detection.

    Args:
        field_names: List of field names
        weights: Optional weights for each field

    Returns:
        List of SearchField objects
    """
    if weights is None:
        weights = {}

    fields = []
    for name in field_names:
        # Auto-detect field type
        if "vector" in name.lower():
            field_type = FieldType.SEMANTIC
        elif "sparse" in name.lower() or "splade" in name.lower():
            field_type = FieldType.SPARSE
        else:
            field_type = FieldType.LEXICAL

        fields.append(SearchField(
            name=name,
            field_type=field_type,
            weight=weights.get(name, 1.0)
        ))

    return fields


def create_basic_retrieval_pipeline(
    locale: Locale,
    es_client: EsClient | None = None,
    synonym_dict: SynonymDict | None = None,
    fusion_method: str = "weighted_sum"
) -> RetrievalPipeline:
    """Create a basic retrieval pipeline with lexical and semantic engines.

    Args:
        locale: Locale for query processing
        es_client: Optional Elasticsearch client
        synonym_dict: Optional synonym dictionary
        fusion_method: Method for fusing results

    Returns:
        Configured RetrievalPipeline
    """
    # Resource manager
    resource_manager = SharedResourceManager()

    # Query processing
    base_processor = BaseQueryProcessor(locale)
    processors: list[QueryProcessor] = [base_processor]

    if synonym_dict:
        synonym_processor = SynonymExpandingProcessor(base_processor, synonym_dict)
        processors.append(synonym_processor)

    semantic_processor = SemanticQueryProcessor(
        base_processor=processors[-1],
        resource_manager=resource_manager,
        model_name=HF.LOCALE_TO_MODEL_NAME.get(locale, HF.JP_SLUKE_MEAN)
    )
    processors.append(semantic_processor)

    query_processor = ProcessorChain(processors)

    # Retrieval engines
    if es_client is None:
        es_client = EsClient()

    engines: list[RetrievalEngine] = [
        LexicalRetrievalEngine(es_client=es_client),
        SemanticRetrievalEngine(es_client=es_client)
    ]

    # Result fusion
    fusion_config = FusionConfig(method=fusion_method, normalization="min_max")
    result_fuser = FlexibleResultFuser(fusion_config)

    return RetrievalPipeline(
        query_processor=query_processor,
        retrieval_engines=engines,
        result_fuser=result_fuser,
        resource_manager=resource_manager
    )


def create_advanced_retrieval_pipeline(
    locale: Locale,
    es_client: EsClient | None = None,
    synonym_dict: SynonymDict | None = None,
    fusion_method: str = "rrf",
    reranker_type: str | None = "dot",
    enable_filtering: bool = False
) -> RetrievalPipeline:
    """Create an advanced retrieval pipeline with reranking and filtering.

    Args:
        locale: Locale for query processing
        es_client: Optional Elasticsearch client
        synonym_dict: Optional synonym dictionary
        fusion_method: Method for fusing results
        reranker_type: Type of reranker ("dot", "colbert", "splade", or None)
        enable_filtering: Whether to enable result filtering

    Returns:
        Configured RetrievalPipeline with post-processing
    """
    # Start with basic pipeline
    pipeline = create_basic_retrieval_pipeline(
        locale=locale,
        es_client=es_client,
        synonym_dict=synonym_dict,
        fusion_method=fusion_method
    )

    # Add reranker if specified
    if reranker_type and pipeline.resource_manager is not None:
        reranker_processor = ResourceAwareRerankerProcessor(
            reranker_type=reranker_type,
            resource_manager=pipeline.resource_manager
        )
        pipeline.add_post_processor(reranker_processor)
        logger.info(f"Added {reranker_type} reranker to pipeline")

    # Add filters if enabled
    if enable_filtering:
        from amazon_product_search.retrieval.processors.reranking import FilterProcessor

        # Example: Filter out results with very low scores
        def score_filter(result, query, config):
            return result.score > 0.1

        filter_processor = FilterProcessor(score_filter, "low_score_filter")
        pipeline.add_post_processor(filter_processor)
        logger.info("Added score filtering to pipeline")

    return pipeline


def create_retriever_with_new_architecture(
    locale: Locale,
    es_client: EsClient | None = None,
    synonym_dict: SynonymDict | None = None,
    enable_reranking: bool = True
) -> Retriever:
    """Create a Retriever instance using the new modular architecture.

    Args:
        locale: Locale for query processing
        es_client: Optional Elasticsearch client
        synonym_dict: Optional synonym dictionary
        enable_reranking: Whether to enable reranking

    Returns:
        Configured Retriever with new architecture
    """
    resource_manager = SharedResourceManager()

    # Optionally preload models for better performance
    models_to_preload = [HF.LOCALE_TO_MODEL_NAME.get(locale, HF.JP_SLUKE_MEAN)]
    if enable_reranking:
        models_to_preload.append(HF.JP_SLUKE_MEAN)  # For dot product reranker

    resource_manager.preload_models(models_to_preload)

    retriever = Retriever(
        locale=locale,
        es_client=es_client,
        synonym_dict=synonym_dict,
        resource_manager=resource_manager,
        use_new_architecture=True
    )

    # Add reranking if enabled
    if enable_reranking:
        reranker_processor = ResourceAwareRerankerProcessor(
            reranker_type="dot",
            resource_manager=resource_manager
        )
        retriever.add_post_processor(reranker_processor)
        logger.info("Added dot product reranker to retriever")

    logger.info(f"Created retriever with new architecture for locale: {locale}")
    return retriever


def create_retriever_with_legacy_architecture(
    locale: Locale,
    es_client: EsClient | None = None,
    query_builder: Any | None = None
) -> Retriever:
    """Create a Retriever instance using the legacy architecture.

    Args:
        locale: Locale for query processing
        es_client: Optional Elasticsearch client
        query_builder: Optional legacy query builder

    Returns:
        Configured Retriever with legacy architecture
    """
    return Retriever(
        locale=locale,
        es_client=es_client,
        query_builder=query_builder,
        use_new_architecture=False
    )


def create_fusion_config(
    method: str = "weighted_sum",
    normalization: str = "min_max",
    weights: dict[str, float] | None = None,
    ranking_constant: int = 60
) -> FusionConfig:
    """Create a fusion configuration.

    Args:
        method: Fusion method ("weighted_sum", "rrf", "borda_count", "max")
        normalization: Score normalization ("min_max", "z_score", "rank_based", "none")
        weights: Engine weights
        ranking_constant: Ranking constant for RRF

    Returns:
        FusionConfig object
    """
    return FusionConfig(
        method=method,
        normalization=normalization,
        weights=weights or {},
        ranking_constant=ranking_constant
    )


# Example usage configurations
RETRIEVAL_CONFIGS: dict[str, dict[str, Any]] = {
    "basic": {
        "description": "Basic lexical + semantic retrieval with min-max fusion",
        "factory": create_basic_retrieval_pipeline,
        "defaults": {"fusion_method": "weighted_sum"}
    },

    "hybrid_rrf": {
        "description": "Hybrid retrieval with RRF fusion and dot product reranking",
        "factory": create_advanced_retrieval_pipeline,
        "defaults": {"fusion_method": "rrf", "reranker_type": "dot"}
    },

    "colbert_advanced": {
        "description": "Advanced retrieval with ColBERT reranking and filtering",
        "factory": create_advanced_retrieval_pipeline,
        "defaults": {"fusion_method": "rrf", "reranker_type": "colbert", "enable_filtering": True}
    },

    "fast_lexical": {
        "description": "Fast lexical-only retrieval (no semantic processing)",
        "factory": lambda locale, **kwargs: create_basic_retrieval_pipeline(
            locale, fusion_method="weighted_sum", **kwargs
        )
    }
}


def create_retrieval_system(
    config_name: str,
    locale: Locale,
    **kwargs
) -> RetrievalPipeline | Retriever:
    """Create a retrieval system using predefined configurations.

    Args:
        config_name: Name of predefined configuration
        locale: Locale for query processing
        **kwargs: Additional arguments to override defaults

    Returns:
        Configured retrieval system

    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name not in RETRIEVAL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(RETRIEVAL_CONFIGS.keys())}")

    config = RETRIEVAL_CONFIGS[config_name]
    factory = config["factory"]

    # Merge defaults with provided kwargs
    params = config.get("defaults", {}).copy()
    params.update(kwargs)

    logger.info(f"Creating retrieval system with config '{config_name}': {config['description']}")
    return factory(locale, **params)
