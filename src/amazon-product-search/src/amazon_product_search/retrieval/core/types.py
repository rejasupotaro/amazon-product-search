from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from amazon_product_search.retrieval.response import Response, Result  # Keep compatibility


class FieldType(Enum):
    """Types of searchable fields."""
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    SPARSE = "sparse"  # For SPLADE-like representations
    HYBRID = "hybrid"


@dataclass
class SearchField:
    """Represents a searchable field with its properties."""
    name: str
    field_type: FieldType
    weight: float = 1.0
    boost: float = 1.0


@dataclass
class ProcessedQuery:
    """Structured representation of a processed query."""
    original: str
    normalized: str
    tokens: list[str] = field(default_factory=list)
    vector: Optional[list[float]] = None
    sparse_vector: Optional[dict[str, float]] = None
    synonyms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    index_name: str
    fields: list[SearchField]
    size: int = 20
    window_size: Optional[int] = None
    enable_synonyms: bool = False
    enable_explain: bool = False
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_fields_by_type(self, field_type: FieldType) -> list[SearchField]:
        """Get fields of a specific type."""
        return [f for f in self.fields if f.field_type == field_type]


@dataclass
class RetrievalResponse:
    """Response from a retrieval engine."""
    results: list[Result]
    total_hits: int
    engine_name: str = ""
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_legacy_response(self) -> Response:
        """Convert to legacy Response format for backward compatibility."""
        return Response(results=self.results, total_hits=self.total_hits)


@dataclass
class FusionConfig:
    """Configuration for result fusion."""
    method: str = "weighted_sum"  # weighted_sum, rrf, borda_count
    normalization: str = "min_max"  # min_max, z_score, rank_based
    weights: dict[str, float] = field(default_factory=dict)
    ranking_constant: int = 60  # For RRF
    metadata: dict[str, Any] = field(default_factory=dict)
