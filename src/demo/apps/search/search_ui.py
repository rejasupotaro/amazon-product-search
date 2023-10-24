from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from amazon_product_search.core.es.response import Response, Result
from amazon_product_search.core.metrics import compute_cosine_similarity


@dataclass
class FormInput:
    query: str | None
    fields: list[str]
    sparse_boost: float
    dense_boost: float
    query_type: str | None
    is_synonym_expansion_enabled: bool
    fuser: str
    fusion_strategy: str
    weighting_strategy: str
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
        sparse_boost = st.number_input("Sparse Boost", value=1.0)
    with columns[1]:
        dense_boost = st.number_input("Dense Boost", value=1.0)

    query_type = st.selectbox(
        "Query Type:",
        options=[
            "combined_fields",
            "cross_fields",
            "best_fields",
            "simple_query_string",
        ],
    )

    is_synonym_expansion_enabled = st.checkbox("enable_synonym_expansion")

    fuser = st.selectbox("fuser:", ["search_engine", "own"])

    fusion_strategy = st.selectbox("fusion_strategy:", ["fuse", "append"])

    weighting_strategy = st.selectbox("weighting_strategy:", ["fixed", "dynamic"])

    reranker_str = st.selectbox("reranker:", ["NoOpReranker", "RandomReranker", "DotReranker"])

    return FormInput(
        query,
        fields,
        sparse_boost,
        dense_boost,
        query_type,
        is_synonym_expansion_enabled,
        fuser=fuser,
        fusion_strategy=fusion_strategy,
        weighting_strategy=weighting_strategy,
        reranker_str=reranker_str,
    )


def draw_response_stats(response: Response, query_vector: np.ndarray, label_dict: dict[str, tuple[str, str]]) -> None:
    rows = []
    for result in response.results:
        sparse_score, dense_score = result.get_scores_in_explanation()
        row = {
            "product_title": result.product["product_title"],
            "total_score": result.score,
            "sparse_score": sparse_score,
            "dense_score": dense_score,
            "label": label_dict.get(result.product["product_id"], ("-", ""))[0],
        }
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


def draw_products(results: list[Result], label_dict: dict[str, tuple[str, str]]) -> None:
    for result in results:
        product = result.product
        header = f"{result.product['product_title']} ({result.score})"
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
            st.write(result.explanation)
