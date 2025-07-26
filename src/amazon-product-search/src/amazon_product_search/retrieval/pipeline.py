import logging
import time
from typing import Any, Sequence

from amazon_product_search.retrieval.core.protocols import (
    QueryProcessor,
    ResourceManager,
    ResultFuser,
    ResultProcessor,
    RetrievalEngine,
)
from amazon_product_search.retrieval.core.types import RetrievalConfig, RetrievalResponse

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Modular retrieval pipeline that orchestrates query processing, retrieval, fusion, and post-processing."""

    def __init__(
        self,
        query_processor: QueryProcessor,
        retrieval_engines: Sequence[RetrievalEngine],
        result_fuser: ResultFuser,
        post_processors: Sequence[ResultProcessor] | None = None,
        resource_manager: ResourceManager | None = None
    ):
        """Initialize retrieval pipeline.

        Args:
            query_processor: Processor for converting raw queries to structured format
            retrieval_engines: List of retrieval engines to query
            result_fuser: Component for fusing multiple retrieval results
            post_processors: Optional list of post-processors (rerankers, filters)
            resource_manager: Optional shared resource manager
        """
        self.query_processor = query_processor
        self.retrieval_engines = list(retrieval_engines)
        self.result_fuser = result_fuser
        self.post_processors = list(post_processors or [])
        self.resource_manager = resource_manager

        logger.info(
            f"Initialized RetrievalPipeline with {len(retrieval_engines)} engines "
            f"and {len(self.post_processors)} post-processors"
        )

    def search(
        self,
        raw_query: str,
        config: RetrievalConfig,
        fusion_weights: dict[str, float] | None = None
    ) -> RetrievalResponse:
        """Execute the full retrieval pipeline.

        Args:
            raw_query: The raw query string from user
            config: Retrieval configuration
            fusion_weights: Optional weights for different engines

        Returns:
            RetrievalResponse: Final fused and processed results
        """
        start_time = time.time()

        logger.debug(f"Starting retrieval pipeline for query: '{raw_query}'")

        # Step 1: Process query
        processed_query = self.query_processor.process(raw_query, config)
        logger.debug(
            f"Processed query: normalized='{processed_query.normalized}', "
            f"tokens={len(processed_query.tokens)}, vector_dim={len(processed_query.vector or [])}"
        )

        # Step 2: Execute retrieval with all compatible engines
        retrieval_responses = []
        for engine in self.retrieval_engines:
            # Check if engine can handle the requested fields
            field_names = [f.name for f in config.fields]
            if not engine.supports_fields(field_names):
                logger.debug(f"Skipping engine {engine.__class__.__name__} - doesn't support fields {field_names}")
                continue

            try:
                response = engine.retrieve(processed_query, config)
                if response.results:  # Only include non-empty responses
                    retrieval_responses.append(response)
                    logger.debug(
                        f"Engine {response.engine_name} returned {len(response.results)} results "
                        f"in {response.processing_time_ms:.2f}ms"
                    )
            except Exception as e:
                logger.error(f"Error in retrieval engine {engine.__class__.__name__}: {e}")
                continue

        if not retrieval_responses:
            logger.warning("No retrieval engines returned results")
            return RetrievalResponse(
                results=[],
                total_hits=0,
                engine_name="pipeline",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Step 3: Fuse results from multiple engines
        if len(retrieval_responses) == 1:
            fused_response = retrieval_responses[0]
            logger.debug("Only one engine returned results, skipping fusion")
        else:
            fused_response = self.result_fuser.fuse(retrieval_responses, fusion_weights)
            logger.debug(f"Fused {len(retrieval_responses)} responses into {len(fused_response.results)} results")

        # Step 4: Apply post-processing (reranking, filtering, etc.)
        final_response = fused_response
        for processor in self.post_processors:
            try:
                processed_response = processor.process(final_response, processed_query, config)
                logger.debug(
                    f"Post-processor {processor.__class__.__name__} processed "
                    f"{len(final_response.results)} -> {len(processed_response.results)} results"
                )
                final_response = processed_response
            except Exception as e:
                logger.error(f"Error in post-processor {processor.__class__.__name__}: {e}")
                continue

        # Step 5: Trim to requested size and update metadata
        if len(final_response.results) > config.size:
            final_response.results = final_response.results[:config.size]

        final_response.processing_time_ms = (time.time() - start_time) * 1000
        final_response.metadata.update({
            "pipeline_stages": {
                "query_processing": True,
                "engines_used": [r.engine_name for r in retrieval_responses],
                "fusion_applied": len(retrieval_responses) > 1,
                "post_processors_applied": len(self.post_processors),
                "final_result_count": len(final_response.results)
            }
        })

        logger.info(
            f"Pipeline completed in {final_response.processing_time_ms:.2f}ms, "
            f"returning {len(final_response.results)} results"
        )
        return final_response



    def add_post_processor(self, processor: ResultProcessor) -> None:
        """Add a post-processor to the pipeline."""
        self.post_processors.append(processor)
        logger.info(f"Added post-processor: {processor.__class__.__name__}")

    def remove_post_processor(self, processor_class: type) -> None:
        """Remove all post-processors of the given class."""
        initial_count = len(self.post_processors)
        self.post_processors = [p for p in self.post_processors if not isinstance(p, processor_class)]
        removed_count = initial_count - len(self.post_processors)
        logger.info(f"Removed {removed_count} post-processors of type {processor_class.__name__}")

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the pipeline configuration."""
        return {
            "query_processor": self.query_processor.__class__.__name__,
            "retrieval_engines": [engine.__class__.__name__ for engine in self.retrieval_engines],
            "result_fuser": self.result_fuser.__class__.__name__,
            "post_processors": [processor.__class__.__name__ for processor in self.post_processors],
            "has_resource_manager": self.resource_manager is not None
        }
