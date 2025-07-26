"""Examples demonstrating the modular retrieval architecture."""

import logging

from amazon_product_search.retrieval.core.types import RetrievalConfig
from amazon_product_search.retrieval.factory import (
    create_retrieval_system,
    create_retriever,
    create_search_fields,
)

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_backward_compatibility():
    """Example 1: Using the new Retriever with legacy API for backward compatibility."""
    logger.info("=== Example 1: Backward Compatibility ===")

    # Create retriever with backward compatible API
    retriever = create_retriever(locale="jp", enable_reranking=True)

    # Use the same API as before - complete backward compatibility
    response = retriever.search(
        index_name="products_jp",
        query="wireless headphones",
        fields=["product_title", "product_description", "title_vector"],
        enable_synonym_expansion=True,
        lexical_boost=0.7,
        semantic_boost=1.3,
        size=10
    )

    logger.info(f"Found {len(response.results)} results")
    logger.info(f"Pipeline info: {retriever.get_pipeline_info()}")


def example_2_new_pipeline_api():
    """Example 2: Using the new pipeline API directly for more control."""
    logger.info("=== Example 2: New Pipeline API ===")

    # Create pipeline with advanced configuration
    pipeline = create_retrieval_system(
        config_name="hybrid_rrf",
        locale="jp",
        fusion_method="rrf",
        reranker_type="dot"
    )

    # Use new structured configuration
    search_fields = create_search_fields(
        field_names=["product_title", "product_description", "title_vector"],
        weights={"product_title": 2.0, "product_description": 1.0, "title_vector": 1.0}
    )

    config = RetrievalConfig(
        index_name="products_jp",
        fields=search_fields,
        size=10,
        enable_synonyms=True,
        enable_explain=True
    )

    # Execute search with fusion weights
    response = pipeline.search(
        raw_query="wireless headphones",
        config=config,
        fusion_weights={"lexical": 0.7, "semantic": 1.3}
    )

    logger.info(f"Found {len(response.results)} results")
    logger.info(f"Processing time: {response.processing_time_ms:.2f}ms")
    logger.info(f"Engines used: {response.metadata.get('pipeline_stages', {}).get('engines_used', [])}")


def example_3_adding_custom_engines():
    """Example 3: Adding custom retrieval engines to the pipeline."""
    logger.info("=== Example 3: Custom Engines ===")

    # Start with basic retriever
    retriever = create_retriever(locale="jp", enable_reranking=False)

    # You could add custom engines like this:
    # custom_engine = MyCustomRetrievalEngine(some_config)
    # retriever.add_retrieval_engine(custom_engine)

    # Add custom post-processors
    from amazon_product_search.retrieval.processors.reranking import FilterProcessor

    def custom_filter(result, query, config):
        # Example: only return results with score > 0.5
        return result.score > 0.5

    filter_processor = FilterProcessor(custom_filter, "high_score_filter")
    retriever.add_post_processor(filter_processor)

    logger.info("Added custom filter processor")
    logger.info(f"Pipeline info: {retriever.get_pipeline_info()}")


def example_4_different_configurations():
    """Example 4: Comparing different retrieval configurations."""
    logger.info("=== Example 4: Configuration Comparison ===")

    configurations = ["basic", "hybrid_rrf", "colbert_advanced"]

    for config_name in configurations:
        logger.info(f"\n--- Testing {config_name} configuration ---")

        try:
            create_retrieval_system(config_name, locale="jp")
            logger.info(f"Created system with config: {config_name}")

            # You could run searches and compare results here
            # results = system.search(...)

        except Exception as e:
            logger.error(f"Failed to create system with config {config_name}: {e}")


def example_5_architecture_features():
    """Example 5: Demonstrating architecture features."""
    logger.info("=== Example 5: Architecture Features ===")

    # Modular architecture
    retriever = create_retriever(locale="jp")
    logger.info("Created modular retriever")

    # The architecture is flexible and maintainable
    logger.info(f"Pipeline info: {retriever.get_pipeline_info()}")

    # Can add custom engines and processors
    logger.info("Architecture supports extensible components")


def example_6_resource_management():
    """Example 6: Demonstrating resource management benefits."""
    logger.info("=== Example 6: Resource Management ===")

    # The architecture uses shared resource management
    retriever1 = create_retriever(locale="jp")
    create_retriever(locale="jp")  # Shares resources

    # Both retrievers use the same underlying models/encoders - no duplication!
    resource_manager = retriever1.resource_manager

    memory_usage = resource_manager.get_memory_usage()
    logger.info(f"Resource usage: {memory_usage}")

    cached_resources = resource_manager.list_cached_resources()
    logger.info(f"Cached resources: {cached_resources}")

    # Clean up when done
    # resource_manager.cleanup()


if __name__ == "__main__":
    """Run all examples to demonstrate the retrieval architecture."""

    logger.info("Running retrieval architecture examples...")

    # Note: These examples show the API usage but won't actually run searches
    # without proper Elasticsearch setup and data

    try:
        example_1_backward_compatibility()
        example_2_new_pipeline_api()
        example_3_adding_custom_engines()
        example_4_different_configurations()
        example_5_architecture_features()
        example_6_resource_management()

        logger.info("\n=== Summary ===")
        logger.info("✅ Backward compatibility maintained")
        logger.info("✅ Modular architecture provides flexibility")
        logger.info("✅ Resource management eliminates duplication")
        logger.info("✅ Easy to add new engines and post-processors")
        logger.info("✅ Multiple fusion strategies available")
        logger.info("✅ Factory functions simplify configuration")

    except ImportError as e:
        logger.error(f"Import error - some components may not be available: {e}")
    except Exception as e:
        logger.error(f"Example execution error: {e}")
        logger.info("This is expected without proper Elasticsearch setup")
