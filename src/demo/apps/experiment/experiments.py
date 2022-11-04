from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Variant:
    name: str
    fields: list[str] = field(default_factory=lambda: ["product_title"])
    top_k: int = 100


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
            Variant(name="title,color", fields=["product_title", "product_color_name"]),  # noqa
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
    "keyword_extraction": ExperimentalSetup(
        index_name="products_jp",
        locale="jp",
        num_queries=5000,
        variants=[
            Variant(name="title", fields=["product_title"]),  # noqa
            Variant(name="title,description", fields=["product_title", "product_description"]),  # noqa
            Variant(name="title,description,bullet_point", fields=["product_title", "product_description", "product_bullet_point"]),  # noqa
            Variant(name="title,yake(description+bullet_point)", fields=["product_title", "product_description_yake"]),  # noqa
            Variant(name="title,position_rank(description+bullet_point)", fields=["product_title", "product_description_position_rank"]),  # noqa
            Variant(name="title,multipartite_rank(description+bullet_point)", fields=["product_title", "product_description_multipartite_rank"]),  # noqa
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
