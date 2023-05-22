from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from amazon_product_search.es.response import Response
from amazon_product_search.metrics import compute_cosine_similarity


@dataclass
class FormInput:
    index_name: str | None
    query: str | None
    fields: list[str]
    sparse_boost: float
    dense_boost: float
    query_type: str | None
    is_synonym_expansion_enabled: bool
    reranker_str: str | None


def draw_input_form(indices: list[str], queries: list[str] | None = None) -> FormInput:
    index_name = st.selectbox("Index:", indices)

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
        sparse_boost = st.number_input("Sparse Boost", value=1.0)
    with columns[1]:
        dense_boost = st.number_input("Dense Boost", value=1.0)

    query_type = st.selectbox(
        "Query Type:",
        options=["combined_fields", "cross_fields", "best_fields", "simple_query_string"],
    )

    is_synonym_expansion_enabled = st.checkbox("enable_synonym_expansion")

    reranker_str = st.selectbox("reranker:", ["NoOpReranker", "RandomReranker", "DotReranker"])

    return FormInput(
        index_name,
        query,
        fields,
        sparse_boost,
        dense_boost,
        query_type,
        is_synonym_expansion_enabled,
        reranker_str,
    )


def draw_response_stats(response: Response, query_vector: np.ndarray):
    rows = []
    for result in response.results:
        row = {"product_title": result.product["product_title"]}

        explanation = result.explanation
        row["total_score"] = explanation["value"]
        sparse_score = 0
        dense_score = None
        if explanation["description"] == "sum of:":
            for child_explanation in explanation["details"]:
                if child_explanation["description"] == "within top k documents":
                    if not dense_score:
                        dense_score = 0
                    dense_score += child_explanation["value"]
                else:
                    sparse_score += child_explanation["value"]
        if dense_score:
            row["sparse_score"] = sparse_score
            row["dense_score"] = dense_score
        else:
            row["sparse_score"] = row["total_score"]
        rows.append(row)

    df = pd.DataFrame(rows)
    with st.expander("Response Stats"):
        st.write(df)

        product_vectors = np.array([result.product["product_vector"] for result in response.results])
        scores = compute_cosine_similarity(query_vector, product_vectors)
        scores_df = pd.DataFrame([{"i": i, "score": score} for i, score in enumerate(scores)])
        fig = px.line(scores_df, x="i", y="score")
        fig.update_layout(title="Cosine Similarity")
        st.plotly_chart(fig, use_container_width=True)
