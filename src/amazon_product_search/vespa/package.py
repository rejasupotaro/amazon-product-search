from vespa.application import ApplicationPackage
from vespa.package import HNSW, Document, Field, FieldSet, RankProfile, Schema

from amazon_product_search.constants import VESPA_DIR


def create_package() -> ApplicationPackage:
    product_document = Document(
        fields=[
            Field(name="product_id", type="string", indexing=["attribute", "summary"]),
            Field(name="product_title", type="string", indexing=["index", "summary"], index="enable-bm25"),
            Field(name="product_bullet_point", type="string", indexing=["index", "summary"], index="enable-bm25"),
            Field(name="product_description", type="string", indexing=["index", "summary"], index="enable-bm25"),
            Field(
                name="product_brand",
                type="string",
                indexing=["attribute", "index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="product_color",
                type="string",
                indexing=["attribute", "index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="product_locale",
                type="string",
                indexing=["attribute", "index", "summary"],
            ),
            Field(
                name="product_vector",
                type="tensor<float>(x[768])",
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="euclidean",
                    max_links_per_node=16,
                    neighbors_to_explore_at_insert=500,
                ),
            ),
        ],
    )
    product_schema = Schema(
        name="product",
        document=product_document,
        fieldsets=[
            FieldSet(
                name="default",
                fields=["product_title", "product_brand", "product_description", "product_brand", "product_color"],
            )
        ],
        rank_profiles=[
            RankProfile(name="random", inherits="default", first_phase="random"),
            RankProfile(name="bm25", inherits="default", first_phase="bm25(product_title) + bm25(product_description)"),
            RankProfile(
                name="native_rank", inherits="default", first_phase="nativeRank(product_title, product_description)"
            ),
            RankProfile(
                name="semantic-similarity",
                inherits="default",
                first_phase="closeness(product_vector)",
                inputs=[("query(query_vector)", "tensor<float>(x[768])")],
            ),
        ],
    )

    app_package = ApplicationPackage(name="amazon", schema=[product_schema])
    return app_package


def dump_config(app_package: ApplicationPackage):
    with open(f"{VESPA_DIR}/services.xml", "w") as f:
        print(app_package.services_to_text, file=f)
    for schema in app_package.schemas:
        with open(f"{VESPA_DIR}/schemas/{schema.name}.sd", "w") as f:
            print(schema.schema_to_text, file=f)
