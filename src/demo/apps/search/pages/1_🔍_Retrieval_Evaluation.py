from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.query_builder import QueryBuilder
from amazon_product_search.es.response import Response, Result
from amazon_product_search.metrics import compute_cosine_similarity, compute_ndcg, compute_recall
from amazon_product_search.nlp.normalizer import normalize_query
from amazon_product_search.reranking.reranker import from_string
from demo.page_config import set_page_config
from demo.utils import load_merged, split_fields

es_client = EsClient()
query_builder = QueryBuilder()


@st.cache_data
def load_dataset() -> dict[str, dict[str, tuple[str, str]]]:
    df = load_merged(locale="jp").to_pandas()
    df = df[df["split"] == "test"]
    query_to_label: dict[str, dict[str, tuple[str, str]]] = {}
    for query, group in df.groupby("query"):
        query_to_label[query] = {}
        for row in group.to_dict("records"):
            query_to_label[query][row["product_id"]] = (row["esci_label"], row["product_title"])
    return query_to_label


def draw_es_query(query: Optional[dict[str, Any]], knn_query: Optional[dict[str, Any]], size: int):
    es_query: dict[str, Any] = {
        "size": size,
    }

    if query:
        es_query["query"] = query

    if knn_query:
        es_query["knn"] = knn_query

    st.write("Elasticsearch Query:")
    st.write(es_query)


def draw_response_stats(response: Response, normalized_query: str):
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

        query_vector = query_builder.encode(normalized_query)
        product_vectors = np.array([result.product["product_vector"] for result in response.results])
        scores = compute_cosine_similarity(query_vector, product_vectors)
        scores_df = pd.DataFrame([{"i": i, "score": score} for i, score in enumerate(scores)])
        fig = px.line(scores_df, x="i", y="score")
        fig.update_layout(title="Cosine Similarity")
        st.plotly_chart(fig, use_container_width=True)


def draw_products(results: list[Result], label_dict: dict[str, str]):
    for result in results:
        product = result.product
        label = {
            "E": "[Exact] ",
            "S": "[Substitute] ",
            "C": "[Complement] ",
            "I": "[Irrelevant] ",
            "-": "",
        }[label_dict.get(product["product_id"], ("-", ""))[0]]
        with st.expander(f"{label}{result.product['product_title']} ({result.score})"):
            st.write(result.product)
            st.write(result.explanation)


def main():
    set_page_config()
    st.write("## Search")

    query_to_label = load_dataset()
    size = 20

    st.write("#### Input")
    with st.form("input"):
        indices = es_client.list_indices()
        index_name = st.selectbox("Index:", indices)

        query = st.selectbox("Query:", query_to_label.keys())
        normalized_query = normalize_query(query)

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
        sparse_fields, dense_fields = split_fields(fields)

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

        reranker = from_string(st.selectbox("reranker:", ["NoOpReranker", "RandomReranker", "DotReranker"]))

        es_query = None
        if sparse_fields:
            es_query = query_builder.build_sparse_search_query(
                query=normalized_query,
                fields=sparse_fields,
                query_type=query_type,
                boost=sparse_boost,
                is_synonym_expansion_enabled=is_synonym_expansion_enabled,
            )
        es_knn_query = None
        if normalized_query and dense_fields:
            # TODO: Should multiple vector fields be handled?
            es_knn_query = query_builder.build_dense_search_query(
                normalized_query, field=dense_fields[0], top_k=size, boost=dense_boost,
            )

        if not st.form_submit_button("Search"):
            return

    st.write("----")

    with st.expander("Query Details", expanded=False):
        st.write("Normalized Query:")
        st.write(normalized_query)

        st.write("Analyzed Query")
        analyzed_query = es_client.analyze(query)
        st.write(analyzed_query)

        draw_es_query(es_query, es_knn_query, size)

    label_dict = query_to_label.get(query, {})
    with st.expander("Labels", expanded=False):
        st.write(label_dict)

    st.write("----")

    st.write("#### Output")
    response = es_client.search(index_name=index_name, query=es_query, knn_query=es_knn_query, size=size, explain=True)
    response.results = reranker.rerank(normalized_query, response.results)
    if not response.results:
        return

    draw_response_stats(response, normalized_query)

    retrieved_ids = [result.product["product_id"] for result in response.results]
    judgements = {product_id: label for product_id, (label, product_title) in label_dict.items()}
    ndcg = compute_ndcg(retrieved_ids, judgements)
    recall = compute_recall(retrieved_ids, judgements.keys())
    st.write(f"{response.total_hits} products found (NDCG: {ndcg}, Recall: {recall})")
    draw_products(response.results, label_dict)


if __name__ == "__main__":
    main()
