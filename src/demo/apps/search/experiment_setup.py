from dataclasses import dataclass, field
from typing import Literal, Optional

from amazon_product_search.core.reranking.reranker import (
    ColBERTReranker,
    DotReranker,
    NoOpReranker,
    RandomReranker,
    Reranker,
    SpladeReranker,
)

SPARSE_FIELDS = [
    "product_title",
    "product_brand",
    "product_color",
    "product_bullet_point",
    "product_description",
]
ALL_FIELDS = [*SPARSE_FIELDS, "product_vector"]
Task = Literal["retrieval", "reranking"]


@dataclass
class RankFusion:
    fuser: Literal["search_engine", "own"] = "search_engine"
    enable_score_normalization: bool = False
    rrf: bool | int = False
    weighting_strategy: Literal["fixed", "dynamic"] = "fixed"


@dataclass
class Variant:
    name: str
    fields: list[str] = field(default_factory=lambda: ["product_title"])
    query_type: str = "combined_fields"
    sparse_boost: float = 1.0
    dense_boost: float = 1.0
    enable_synonym_expansion: bool = False
    top_k: int = 100
    rank_fusion: RankFusion = field(default_factory=lambda: RankFusion())
    reranker: Reranker = field(default_factory=lambda: NoOpReranker())


@dataclass
class ExperimentSetup:
    task: Task
    num_queries: Optional[int] = None
    variants: list[Variant] = field(default_factory=list)


EXPERIMENTS = {
    "query_types": ExperimentSetup(
        task="retrieval",
        num_queries=5000,
        variants=[
            Variant(name="best_fields", fields=SPARSE_FIELDS, query_type="best_fields"),
            Variant(name="cross_fields", fields=SPARSE_FIELDS, query_type="cross_fields"),
            Variant(
                name="combined_fields",
                fields=SPARSE_FIELDS,
                query_type="combined_fields",
            ),
            Variant(
                name="simple_query_string",
                fields=SPARSE_FIELDS,
                query_type="simple_query_string",
            ),
        ],
    ),
    "different_fields": ExperimentSetup(
        task="retrieval",
        num_queries=5000,
        variants=[
            Variant(name="title", fields=["product_title"]),
            Variant(
                name="title,description",
                fields=["product_title", "product_description"],
            ),
            Variant(
                name="title,bullet_point",
                fields=["product_title", "product_bullet_point"],
            ),
            Variant(name="title,brand", fields=["product_title", "product_brand"]),
            Variant(name="title,color", fields=["product_title", "product_color"]),
            Variant(name="all", fields=SPARSE_FIELDS),
        ],
    ),
    "different_weights": ExperimentSetup(
        task="retrieval",
        num_queries=5000,
        variants=[
            Variant(name="title", fields=["product_title"]),
            Variant(
                name="title^1,bullet_point^1",
                fields=["product_title^1", "product_bullet_point^1"],
            ),
            Variant(
                name="title^2,bullet_point^1",
                fields=["product_title^2", "product_bullet_point^1"],
            ),
            Variant(
                name="title^5,bullet_point^1",
                fields=["product_title^5", "product_bullet_point^1"],
            ),
            Variant(
                name="title^10,bullet_point^1",
                fields=["product_title^10", "product_bullet_point^1"],
            ),
        ],
    ),
    "synonym_expansion": ExperimentSetup(
        task="retrieval",
        num_queries=5000,
        variants=[
            Variant(name="title", fields=["product_title"], enable_synonym_expansion=False),
            Variant(
                name="title,brand,color",
                fields=["product_title", "product_brand", "product_color"],
            ),
            Variant(
                name="title + query expansion",
                fields=["product_title"],
                enable_synonym_expansion=True,
            ),
            Variant(
                name="title,brand,color + query expansion",
                fields=["product_title", "product_brand", "product_color"],
                enable_synonym_expansion=True,
            ),
        ],
    ),
    "sparse_vs_dense": ExperimentSetup(
        task="retrieval",
        num_queries=10,
        variants=[
            Variant(name="sparse only", fields=SPARSE_FIELDS),
            Variant(name="dense only", fields=["product_vector"]),
            Variant(
                name="sparse * 1.0 + dense * 1.0",
                fields=ALL_FIELDS,
                dense_boost=1.0,
            ),
            Variant(
                name="sparse * 1.0 + dense * 5.0",
                fields=ALL_FIELDS,
                dense_boost=5.0,
            ),
            Variant(
                name="sparse * 1.0 + dense * 10.0",
                fields=ALL_FIELDS,
                dense_boost=10.0,
            ),
            Variant(
                name="sparse * 1.0 + dense * 20.0",
                fields=ALL_FIELDS,
                dense_boost=20.0,
            ),
            Variant(
                name="Min-Max Normalization",
                fields=ALL_FIELDS,
                rank_fusion=RankFusion(
                    fuser="own",
                    enable_score_normalization=True,
                ),
            ),
            Variant(
                name="RRF (10)",
                fields=ALL_FIELDS,
                rank_fusion=RankFusion(
                    fuser="own",
                    rrf=10,
                ),
            ),
            Variant(
                name="RRF (60)",
                fields=ALL_FIELDS,
                rank_fusion=RankFusion(
                    fuser="own",
                    rrf=60,
                ),
            ),
        ],
    ),
    "weighting_strategy": ExperimentSetup(
        task="retrieval",
        num_queries=1000,
        variants=[
            Variant(
                name="FixedWeighting (0.5, 0.5)",
                fields=ALL_FIELDS,
                sparse_boost=0.5,
                dense_boost=0.5,
                rank_fusion=RankFusion(
                    fuser="own",
                    enable_score_normalization=True,
                    weighting_strategy="fixed",
                ),
            ),
            Variant(
                name="FixedWeighting (0.8, 0.2)",
                fields=ALL_FIELDS,
                sparse_boost=0.8,
                dense_boost=0.2,
                rank_fusion=RankFusion(
                    fuser="own",
                    enable_score_normalization=True,
                    weighting_strategy="fixed",
                ),
            ),
            Variant(
                name="DynamicWeighting",
                fields=ALL_FIELDS,
                rank_fusion=RankFusion(
                    fuser="own",
                    enable_score_normalization=True,
                    weighting_strategy="dynamic",
                ),
            ),
        ],
    ),
    "reranking": ExperimentSetup(
        task="reranking",
        num_queries=500,
        variants=[
            Variant(
                name="title,bullet_point",
                fields=["product_title", "product_description", "product_bullet_point"],
                enable_synonym_expansion=True,
                reranker=NoOpReranker(),
            ),
            Variant(
                name="title,bullet_point + Random",
                fields=["product_title", "product_description", "product_bullet_point"],
                enable_synonym_expansion=True,
                reranker=RandomReranker(),
            ),
            Variant(
                name="title,bullet_point + SBERT",
                fields=["product_title", "product_description", "product_bullet_point"],
                enable_synonym_expansion=True,
                reranker=DotReranker(),
            ),
            Variant(
                name="title,bullet_point + ColBERT",
                fields=["product_title", "product_description", "product_bullet_point"],
                enable_synonym_expansion=True,
                reranker=ColBERTReranker(),
            ),
            Variant(
                name="title,bullet_point + SPLADE",
                fields=["product_title", "product_description", "product_bullet_point"],
                enable_synonym_expansion=True,
                reranker=SpladeReranker(),
            ),
        ],
    ),
}
