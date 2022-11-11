from dataclasses import dataclass, field
from typing import Optional

from amazon_product_search.reranking.reranker import NoOpReranker, RandomReranker, Reranker, SentenceBERTReranker


@dataclass
class Variant:
    name: str
    fields: list[str] = field(default_factory=lambda: ["product_title"])
    enable_synonym_expansion: bool = False
    top_k: int = 100
    reranker: Reranker = field(default_factory=lambda: NoOpReranker())


@dataclass
class ExperimentalSetup:
    index_name: str
    locale: str
    variants: list[Variant]
    num_queries: Optional[int] = None


# fmt: off
EXPERIMENTS = {
    "different_fields": ExperimentalSetup(
        index_name="products_all_jp",
        locale="jp",
        num_queries=5000,
        variants=[
            Variant(name="title", fields=["product_title"]),  # noqa
            Variant(name="title,description", fields=["product_title", "product_description"]),  # noqa
            Variant(name="title,bullet_point", fields=["product_title", "product_bullet_point"]),  # noqa
            Variant(name="title,brand", fields=["product_title", "product_brand"]),  # noqa
            Variant(name="title,color", fields=["product_title", "product_color"]),  # noqa
        ],
    ),
    "different_weights": ExperimentalSetup(
        index_name="products_all_jp",
        locale="jp",
        num_queries=5000,
        variants=[
            Variant(name="title", fields=["product_title"]),  # noqa
            Variant(name="title^1,bullet_point^1", fields=["product_title^1", "product_bullet_point^1"]),  # noqa
            Variant(name="title^2,bullet_point^1", fields=["product_title^2", "product_bullet_point^1"]),  # noqa
            Variant(name="title^5,bullet_point^1", fields=["product_title^5", "product_bullet_point^1"]),  # noqa
            Variant(name="title^10,bullet_point^1", fields=["product_title^10", "product_bullet_point^1"]),  # noqa
        ],
    ),
    "synonym_expansion": ExperimentalSetup(
        index_name="products_all_jp",
        locale="jp",
        num_queries=5000,
        variants=[
            Variant(name="title", fields=["product_title"], enable_synonym_expansion=False),  # noqa
            Variant(name="title,description", fields=["product_title", "product_description"], enable_synonym_expansion=False),  # noqa
            Variant(name="title,bullet_point", fields=["product_title", "product_bullet_point"], enable_synonym_expansion=False),  # noqa
            Variant(name="title,brand", fields=["product_title", "product_brand"]),  # noqa
            Variant(name="title,color", fields=["product_title", "product_color"]),  # noqa
            Variant(name="query expansion + title", fields=["product_title"], enable_synonym_expansion=True),  # noqa
            Variant(name="query expansion + title,description", fields=["product_title", "product_description"], enable_synonym_expansion=True),  # noqa
            Variant(name="query expansion + title,bullet_point", fields=["product_title", "product_bullet_point"], enable_synonym_expansion=True),  # noqa
            Variant(name="query expansion + title,brand", fields=["product_title", "product_brand"], enable_synonym_expansion=True),  # noqa
            Variant(name="query expansion + title,color", fields=["product_title", "product_color"], enable_synonym_expansion=True),  # noqa
        ],
    ),
    "keyword_extraction": ExperimentalSetup(
        index_name="products_ke_jp",
        locale="jp",
        num_queries=5000,
        variants=[
            Variant(name="title", fields=["product_title"]),  # noqa
            Variant(name="title,description,bullet_point", fields=["product_title", "product_description", "product_bullet_point"]),  # noqa
            Variant(name="title,yake(description+bullet_point)", fields=["product_title", "product_description_yake"]),  # noqa
            Variant(name="title,position_rank(description+bullet_point)", fields=["product_title", "product_description_position_rank"]),  # noqa
            Variant(name="title,multipartite_rank(description+bullet_point)", fields=["product_title", "product_description_multipartite_rank"]),  # noqa
        ],
    ),
    "reranking": ExperimentalSetup(
        index_name="products_all_jp",
        locale="jp",
        num_queries=500,
        variants=[
            Variant(name="title,bullet_point", fields=["product_title", "product_description"], reranker=NoOpReranker()),  # noqa
            Variant(name="title,bullet_point + random reranker", fields=["product_title", "product_description"], reranker=RandomReranker()),  # noqa
            Variant(name="title,bullet_point + sbert reranker", fields=["product_title", "product_description"], reranker=SentenceBERTReranker()),  # noqa
        ],
    ),
    "sparse_vs_dense": ExperimentalSetup(
        index_name="products_dense_jp",
        locale="jp",
        num_queries=50,
        variants=[
            Variant(name="sparse", fields=["product_title"]),  # noqa
            Variant(name="dense", fields=["product_vector"]),  # noqa
            Variant(name="hybrid", fields=["product_title", "product_vector"]),  # noqa
        ],
    ),
}
# fmt: on
