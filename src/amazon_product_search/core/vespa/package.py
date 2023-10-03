from vespa.application import ApplicationPackage
from vespa.package import (
    HNSW,
    Document,
    Field,
    FieldSet,
    OnnxModel,
    RankProfile,
    Schema,
)


def create_package() -> ApplicationPackage:
    product_document = Document(
        fields=[
            Field(name="product_id", type="string", indexing=["attribute", "summary"]),
            Field(
                name="product_title",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="product_bullet_point",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="product_description",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
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
                    distance_metric="angular",
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
                fields=[
                    "product_title",
                    "product_brand",
                    "product_description",
                    "product_brand",
                    "product_color",
                ],
            )
        ],
        models=[
            OnnxModel(
                model_name="sbert",
                model_file_path="./models/sbert.onnx",
                inputs={
                    "input_ids": "input_ids",
                    "token_type_ids": "token_type_ids",
                    "attention_mask": "attention_mask",
                },
                outputs={"output_0": "output_0", "output_1": "output_1"},
            ),
        ],
        rank_profiles=[
            RankProfile(name="random", inherits="default", first_phase="random"),
            RankProfile(
                name="bm25",
                inherits="default",
                first_phase="bm25(product_title) + bm25(product_description)",
            ),
            RankProfile(
                name="native_rank",
                inherits="default",
                first_phase="nativeRank(product_title, product_description)",
            ),
            RankProfile(
                name="semantic_similarity",
                inherits="default",
                first_phase="closeness(product_vector)",
                inputs=[("query(query_vector)", "tensor<float>(x[768])")],
            ),
        ],
    )

    app_package = ApplicationPackage(
        name="amazon",
        schema=[product_schema],
    )
    return app_package
