from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from amazon_product_search.metrics import compute_cosine_similarity
from amazon_product_search.retrieval.response import Response, Result


@dataclass
class FormInput:
    query: str | None
    fields: list[str]
    lexical_boost: float
    semantic_boost: float
    enable_synonym_expansion: bool
    size: int
    window_size: int
    combination_method: Literal["sum", "max", "append"]
    score_transformation_method: Literal["min_max", "rrf"] | None
    reranker_str: str | None


def draw_input_form(queries: list[str] | None = None) -> FormInput:
    query = st.selectbox("Query:", queries) if queries else st.text_input("Query:")

    fields = st.multiselect(
        "Fields:",
        options=[
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_vector",
        ],
        default=["product_title"],
    )

    columns = st.columns(2)
    with columns[0]:
        lexical_boost = st.number_input("Lexical Boost", value=1.0)
    with columns[1]:
        semantic_boost = st.number_input("Semantic Boost", value=1.0)

    enable_synonym_expansion = st.checkbox("enable_synonym_expansion")

    size = st.number_input("size", value=100)

    window_size = st.number_input("window_size", value=100)

    combination_method = st.selectbox("combination_method:", ["sum", "max", "append"])

    score_transformation_method = st.selectbox("score_transformation_method:", ["min_max", "rrf", "borda"])

    reranker_str = st.selectbox("reranker:", ["NoOpReranker", "RandomReranker", "DotReranker"])

    return FormInput(
        query,
        fields,
        lexical_boost,
        semantic_boost,
        enable_synonym_expansion,
        size=size,
        window_size=window_size,
        combination_method=combination_method,
        score_transformation_method=score_transformation_method,
        reranker_str=reranker_str,
    )


def draw_response_stats(response: Response, query_vector: np.ndarray, label_dict: dict[str, tuple[str, str]]) -> None:
    rows = []
    for result in response.results:
        lexical_score, semantic_score = result.get_scores_in_explanation()
        row = {
            "product_title": result.product["product_title"],
            "total_score": result.score,
            "lexical_score": lexical_score,
            "semantic_score": semantic_score,
            "label": label_dict.get(result.product["product_id"], ("-", ""))[0],
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    with st.expander("Response Stats"):
        st.write(df)

        rows = []
        for i, row in enumerate(df.to_dict(orient="records")):
            rows.append(
                {
                    "retrieval": "lexical",
                    "rank": i,
                    "score": row["lexical_score"],
                }
            )
            rows.append(
                {
                    "retrieval": "semantic",
                    "rank": i,
                    "score": row["semantic_score"],
                }
            )
        scores_df = pd.DataFrame(rows)
        fig = px.bar(scores_df, x="rank", y="score", color="retrieval")
        fig.update_layout(title="Scores by Rank")
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            fig = px.histogram(scores_df, x="score", color="retrieval", barmode="overlay")
            fig.update_layout(title="Histogram of Scores by Retrieval")
            st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            df["label"] = df["label"].astype("category")
            fig = px.scatter(df, x="lexical_score", y="semantic_score", color="label")
            fig.update_layout(title="Scores and Labels")
            st.plotly_chart(fig, use_container_width=True)

        if not response.results:
            return
        if not response.results[0].product.get("product_vector"):
            return
        product_vectors = np.array([result.product["product_vector"] for result in response.results])
        scores = compute_cosine_similarity(query_vector, product_vectors)
        scores_df = pd.DataFrame([{"i": i, "score": score} for i, score in enumerate(scores)])
        fig = px.line(scores_df, x="i", y="score")
        fig.update_layout(title="Cosine Similarity")
        st.plotly_chart(fig, use_container_width=True)


def draw_product(result: Result, label_dict: dict[str, tuple[str, str]]) -> None:
    product = result.product

    image_url = product.get("image_url")
    if image_url:
        st.image(image_url, width=200)

    header = f"{product['product_title']} ({round(result.score, 4)})"
    if label_dict:
        label = {
            "E": "[Exact] ",
            "S": "[Substitute] ",
            "C": "[Complement] ",
            "I": "[Irrelevant] ",
            "-": "",
        }[label_dict.get(product["product_id"], ("-", ""))[0]]
        header = f"{label}{header}"

    with st.expander(header):
        st.write(result.product)

    if result.explanation:
        with st.expander("Explanation"):
            st.write(result.explanation)


def draw_products(results: list[Result], label_dict: dict[str, tuple[str, str]]) -> None:
    for result in results:
        st.write("---")
        draw_product(result, label_dict)
