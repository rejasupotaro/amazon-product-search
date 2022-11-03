from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Variant:
    name: str
    is_sparse_enabled: bool = True
    is_dense_enabled: bool = False
    fields: list[str] = field(default_factory=lambda: ["product_title"])
    top_k: int = 100


@dataclass
class ExperimentalSetup:
    locale: str
    variants: list[Variant]
    num_queries: Optional[int] = None

    @property
    def index_name(self) -> str:
        return f"products_{self.locale}"


# fmt: off
EXPERIMENTS = {
    "different_fields": ExperimentalSetup(
        locale="jp",
        num_queries=1000,
        variants=[
            Variant(name="title", fields=["product_title"]),  # noqa
            Variant(name="title,description", fields=["product_title", "product_description"]),  # noqa
            Variant(name="title,bullet_point", fields=["product_title", "product_bullet_point"]),  # noqa
            Variant(name="title,brand", fields=["product_title", "product_brand"]),  # noqa
            Variant(name="title,color", fields=["product_title", "product_color_name"]),  # noqa
        ],
    ),
    "different_weights": ExperimentalSetup(
        locale="jp",
        num_queries=1000,
        variants=[
            Variant(name="title", fields=["product_title"]),  # noqa
            Variant(name="title^2,description", fields=["product_title^2", "product_description"]),  # noqa
            Variant(name="title^3,description", fields=["product_title^3", "product_description"]),  # noqa
            Variant(name="title^5,description", fields=["product_title^5", "product_description"]),  # noqa
            Variant(name="title^10,description", fields=["product_title^10", "product_description"]),  # noqa
        ],
    ),
    "keyword_extraction": ExperimentalSetup(
        locale="jp",
        num_queries=1000,
        variants=[
            Variant(name="title", fields=["product_title"]),  # noqa
            Variant(name="title,description", fields=["product_title", "product_description"]),  # noqa
            Variant(name="title,description,bullet_point", fields=["product_title", "product_description", "product_bullet_point"]),  # noqa
            Variant(name="title,pke(description+bullet_point)", fields=["product_title", "product_description_pke"]),  # noqa
            Variant(name="title,position_rank(description+bullet_point)", fields=["product_title", "product_description_position_rank"]),  # noqa
            Variant(name="title,multipartite_rank(description+bullet_point)", fields=["product_title", "product_description_multipartite_rank"]),  # noqa
        ],
    ),
    "sparse_vs_dense": ExperimentalSetup(
        locale="jp",
        num_queries=1000,
        variants=[
            Variant(name="sparse", is_sparse_enabled=True, is_dense_enabled=False),  # noqa
            Variant(name="dense", is_sparse_enabled=False, is_dense_enabled=True),  # noqa
            Variant(name="hybrid", is_sparse_enabled=True, is_dense_enabled=True),  # noqa
        ],
    ),
}
# fmt: on
