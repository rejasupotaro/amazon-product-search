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
from amazon_product_search.core.source import Locale

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
class Variant:
    name: str
    fields: list[str] = field(default_factory=lambda: ["product_title"])
    query_type: str = "combined_fields"
    dense_boost: float = 1.0
    enable_synonym_expansion: bool = False
    top_k: int = 100
    rrf: bool | int = False
    reranker: Reranker = field(default_factory=lambda: NoOpReranker())


@dataclass
class ExperimentalSetup:
    index_name: str
    locale: Locale
    task: Task
    num_queries: Optional[int] = None
    variants: list[Variant] = field(default_factory=list)


EXPERIMENTS = {
    "query_types": ExperimentalSetup(
        index_name="products_jp",
        locale="jp",
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
    "different_fields": ExperimentalSetup(
        index_name="products_jp",
        locale="jp",
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
    "different_weights": ExperimentalSetup(
        index_name="products_jp",
        locale="jp",
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
    "synonym_expansion": ExperimentalSetup(
        index_name="products_jp",
        locale="jp",
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
    "sparse_vs_dense": ExperimentalSetup(
        index_name="products_jp",
        locale="jp",
        task="retrieval",
        num_queries=5000,
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
                name="RRF (10)",
                fields=ALL_FIELDS,
                rrf=10,
            ),
            Variant(
                name="RRF (60)",
                fields=ALL_FIELDS,
                rrf=60,
            ),
        ],
    ),
    "reranking": ExperimentalSetup(
        index_name="products_jp",
        locale="jp",
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
