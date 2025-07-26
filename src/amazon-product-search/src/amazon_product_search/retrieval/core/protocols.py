from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from amazon_product_search.retrieval.core.types import ProcessedQuery, RetrievalConfig, RetrievalResponse


@runtime_checkable
class QueryProcessor(Protocol):
    """Abstract interface for query processing components."""

    def process(self, raw_query: str, config: RetrievalConfig) -> ProcessedQuery:
        """Process a raw query string into a structured format.

        Args:
            raw_query: The original query string
            config: Retrieval configuration

        Returns:
            ProcessedQuery: Structured query representation
        """
        ...


@runtime_checkable
class RetrievalEngine(Protocol):
    """Abstract interface for retrieval engines."""

    def retrieve(self, query: ProcessedQuery, config: RetrievalConfig) -> RetrievalResponse:
        """Retrieve documents for a given query.

        Args:
            query: Processed query
            config: Retrieval configuration

        Returns:
            RetrievalResponse: Retrieved results with metadata
        """
        ...

    def supports_fields(self, fields: list[str]) -> bool:
        """Check if this engine can handle the given field types.

        Args:
            fields: List of field names to check

        Returns:
            bool: True if engine supports these fields
        """
        ...


@runtime_checkable
class ResultFuser(Protocol):
    """Abstract interface for combining multiple retrieval results."""

    def fuse(self, responses: list[RetrievalResponse], weights: dict[str, float] | None = None) -> RetrievalResponse:
        """Fuse multiple retrieval responses into a single response.

        Args:
            responses: List of retrieval responses to combine
            weights: Optional weights for each response

        Returns:
            RetrievalResponse: Combined results
        """
        ...


@runtime_checkable
class ResultProcessor(Protocol):
    """Abstract interface for post-processing retrieval results."""

    def process(self, response: RetrievalResponse, query: ProcessedQuery, config: RetrievalConfig) -> RetrievalResponse:
        """Process retrieval results (reranking, filtering, etc.).

        Args:
            response: Original retrieval response
            query: Processed query
            config: Retrieval configuration

        Returns:
            RetrievalResponse: Processed results
        """
        ...


class ResourceManager(ABC):
    """Abstract base class for managing shared resources like models and encoders."""

    @abstractmethod
    def get_encoder(self, model_name: str) -> Any:
        """Get an encoder instance for the given model."""
        ...

    @abstractmethod
    def get_tokenizer(self, model_name: str) -> Any:
        """Get a tokenizer instance for the given model."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        ...
