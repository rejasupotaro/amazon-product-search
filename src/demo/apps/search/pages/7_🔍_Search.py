from typing import Any, Optional

import pandas as pd
import streamlit as st

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.query_builder import QueryBuilder
from amazon_product_search.es.response import Response, Result
from amazon_product_search.nlp.normalizer import normalize_query
from amazon_product_search.reranking.reranker import from_string
from demo.page_config import set_page_config
from demo.utils import split_fields

es_client = EsClient()
query_builder = QueryBuilder()


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


def draw_response_stats(response: Response):
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
        rows.append(row)

    df = pd.DataFrame(rows)
    with st.expander("Response Stats"):
        st.write(df)
        st.write(df.describe())


def draw_products(results: list[Result]):
    for result in results:
        with st.expander(f"{result.product['product_title']} ({result.score})"):
            st.write(result.product)
            st.write(result.explanation)


def main():
    set_page_config()
    st.write("## Search")

    size = 20

    st.write("#### Input")
    with st.form("input"):
        indices = es_client.list_indices()
        index_name = st.selectbox("Index:", indices)

        query = st.text_input("Query:")
        normalized_query = normalize_query(query)

        selected_fields = st.multiselect(
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
        sparse_fields, dense_fields = split_fields(selected_fields)

        query_type = st.selectbox(
            "Query Type:",
            options=["cross_fields", "best_fields", "combined_fields", "simple_query_string"],
        )

        is_synonym_expansion_enabled = st.checkbox("enable_synonym_expansion")

        reranker = from_string(st.selectbox("reranker:", ["NoOpReranker", "RandomReranker", "DotReranker"]))

        es_query = None
        if sparse_fields:
            es_query = query_builder.build_sparse_search_query(
                query=normalized_query,
                fields=sparse_fields,
                query_type=query_type,
                is_synonym_expansion_enabled=is_synonym_expansion_enabled,
            )
        es_knn_query = None
        if normalized_query and dense_fields:
            # TODO: Should multiple vector fields be handled?
            es_knn_query = query_builder.build_dense_search_query(normalized_query, field=dense_fields[0], top_k=size)

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

    st.write("----")

    st.write("#### Output")
    response = es_client.search(index_name=index_name, query=es_query, knn_query=es_knn_query, size=size, explain=True)
    response.results = reranker.rerank(normalized_query, response.results)
    draw_response_stats(response)
    st.write(f"{response.total_hits} products found")
    draw_products(response.results)


if __name__ == "__main__":
    main()
