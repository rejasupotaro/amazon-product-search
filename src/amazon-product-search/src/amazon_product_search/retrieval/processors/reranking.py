import logging
from typing import cast

from amazon_product_search.reranking.reranker import ColBERTReranker, Reranker, SpladeReranker
from amazon_product_search.retrieval.core.protocols import ResourceManager, ResultProcessor
from amazon_product_search.retrieval.core.types import ProcessedQuery, RetrievalConfig, RetrievalResponse
from amazon_product_search.retrieval.response import Result

logger = logging.getLogger(__name__)


class RerankerProcessor(ResultProcessor):
    """Adapter that wraps existing reranker implementations as result processors."""

    def __init__(self, reranker: Reranker, name: str | None = None):
        """Initialize reranker processor.

        Args:
            reranker: The reranker implementation to wrap
            name: Optional name for logging purposes
        """
        self.reranker = reranker
        self.name = name or reranker.__class__.__name__

    def process(self, response: RetrievalResponse, query: ProcessedQuery, config: RetrievalConfig) -> RetrievalResponse:
        """Process results by reranking them.

        Args:
            response: Original retrieval response
            query: Processed query
            config: Retrieval configuration

        Returns:
            RetrievalResponse: Response with reranked results
        """
        if not response.results:
            return response

        logger.debug(f"Reranking {len(response.results)} results using {self.name}")

        try:
            # Rerank the results
            reranked_results = self.reranker.rerank(query.original, response.results)

            # Create new response with reranked results
            reranked_response = RetrievalResponse(
                results=reranked_results,
                total_hits=response.total_hits,
                engine_name=response.engine_name,
                processing_time_ms=response.processing_time_ms,
                metadata=response.metadata.copy()
            )

            # Add reranking metadata
            reranked_response.metadata["reranked_by"] = self.name
            reranked_response.metadata["original_order_changed"] = (
                [r.product["product_id"] for r in response.results] !=
                [r.product["product_id"] for r in reranked_results]
            )

            logger.debug(f"Reranking completed with {self.name}")
            return reranked_response

        except Exception as e:
            logger.error(f"Error during reranking with {self.name}: {e}")
            # Return original response on error
            return response


class ResourceAwareRerankerProcessor(ResultProcessor):
    """Reranker processor that uses shared resource manager."""

    def __init__(
        self,
        reranker_type: str,
        resource_manager: ResourceManager,
        model_name: str | None = None,
        **kwargs
    ):
        """Initialize resource-aware reranker processor.

        Args:
            reranker_type: Type of reranker ("dot", "colbert", "splade")
            resource_manager: Shared resource manager
            model_name: Optional model name override
            **kwargs: Additional arguments for reranker
        """
        self.reranker_type = reranker_type
        self.resource_manager = resource_manager
        self.model_name = model_name
        self.kwargs = kwargs
        self._reranker: Reranker | None = None

    @property
    def reranker(self) -> Reranker:
        """Lazy-load reranker instance."""
        if self._reranker is None:
            if self.reranker_type == "dot":
                from amazon_product_search.constants import HF
                model_name = self.model_name or HF.JP_SLUKE_MEAN

                # Create a resource-managed dot reranker
                self._reranker = ResourceManagedDotReranker(
                    self.resource_manager, model_name, **self.kwargs
                )
            elif self.reranker_type == "colbert":
                from amazon_product_search.constants import HF
                self._reranker = ColBERTReranker(
                    model_filepath=self.model_name or HF.JP_COLBERT,
                    **self.kwargs
                )
            elif self.reranker_type == "splade":
                from amazon_product_search.constants import HF
                self._reranker = SpladeReranker(
                    model_filepath=self.model_name or HF.JP_SPLADE,
                    **self.kwargs
                )
            else:
                raise ValueError(f"Unknown reranker type: {self.reranker_type}")

        if self._reranker is None:
            raise RuntimeError(f"Failed to initialize reranker of type: {self.reranker_type}")

        # Cast to ensure mypy knows it's not None after the check
        return cast(Reranker, self._reranker)

    def process(self, response: RetrievalResponse, query: ProcessedQuery, config: RetrievalConfig) -> RetrievalResponse:
        """Process results by reranking them."""
        processor = RerankerProcessor(self.reranker, f"{self.reranker_type}_reranker")
        return processor.process(response, query, config)


class ResourceManagedDotReranker(Reranker):
    """Dot product reranker that uses shared resource manager."""

    def __init__(self, resource_manager: ResourceManager, model_name: str, batch_size: int = 8):
        self.resource_manager = resource_manager
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        """Rerank results using dot product similarity."""
        if not query or not results:
            return results

        import torch

        # Get model and tokenizer from resource manager
        model = self.resource_manager.get_encoder(self.model_name)
        tokenizer = self.resource_manager.get_tokenizer(self.model_name)

        with torch.no_grad():
            # Tokenize query
            query_tokens = tokenizer(
                [query],
                add_special_tokens=True,
                padding="longest",
                truncation="longest_first",
                return_attention_mask=True,
                return_tensors="pt",
            )

            # Tokenize product titles
            product_titles = [result.product["product_title"] for result in results]
            product_tokens = tokenizer(
                product_titles,
                add_special_tokens=True,
                padding="longest",
                truncation="longest_first",
                return_attention_mask=True,
                return_tensors="pt",
            )

            # Get embeddings
            query_emb = model(**query_tokens, return_dict=True).last_hidden_state[:, 0]  # CLS token
            product_embs = model(**product_tokens, return_dict=True).last_hidden_state[:, 0]  # CLS token

            # Compute similarities
            query_emb_repeated = query_emb.repeat(len(results), 1)
            scores = torch.diagonal(torch.mm(query_emb_repeated, product_embs.transpose(0, 1)))

        # Sort results by score
        results_with_scores = list(zip(results, scores.numpy(), strict=True))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)

        return [result for result, _ in results_with_scores]


class FilterProcessor(ResultProcessor):
    """Generic result filter processor."""

    def __init__(self, filter_func, name: str = "filter"):
        """Initialize filter processor.

        Args:
            filter_func: Function that takes (result, query, config) and returns bool
            name: Name for logging purposes
        """
        self.filter_func = filter_func
        self.name = name

    def process(self, response: RetrievalResponse, query: ProcessedQuery, config: RetrievalConfig) -> RetrievalResponse:
        """Filter results based on the provided function."""
        if not response.results:
            return response

        original_count = len(response.results)
        filtered_results = [
            result for result in response.results
            if self.filter_func(result, query, config)
        ]

        filtered_response = RetrievalResponse(
            results=filtered_results,
            total_hits=len(filtered_results),  # Update total hits
            engine_name=response.engine_name,
            processing_time_ms=response.processing_time_ms,
            metadata=response.metadata.copy()
        )

        filtered_response.metadata["filtered_by"] = self.name
        filtered_response.metadata["original_count"] = original_count
        filtered_response.metadata["filtered_count"] = len(filtered_results)

        logger.debug(f"Filtered {original_count} -> {len(filtered_results)} results using {self.name}")
        return filtered_response
